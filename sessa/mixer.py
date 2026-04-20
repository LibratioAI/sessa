import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False

try:
    if _FLASH_ATTN_AVAILABLE:
        try:
            from flash_attn import flash_attn_with_kvcache  
        except Exception:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache  
        _FLASH_KVCACHE_AVAILABLE = True
    else:
        flash_attn_with_kvcache = None
        _FLASH_KVCACHE_AVAILABLE = False
except Exception:
    flash_attn_with_kvcache = None
    _FLASH_KVCACHE_AVAILABLE = False
 
def causal_softmax(scores_fb: torch.Tensor, causal_mask_fb: torch.Tensor) -> torch.Tensor:
    # scores_fb: (B,T,T), causal_mask_fb: (T,T) boolean where True = masked
    B, T, _ = scores_fb.shape
    scores = scores_fb.masked_fill(causal_mask_fb, float("-inf"))

    if T > 0:
        scores = scores.clone()
        scores[:, 0, 0] = 0.0

    alpha = torch.softmax(scores, dim=-1)
    alpha = alpha.masked_fill(causal_mask_fb, 0.0)

    if T > 0:
        alpha[:, 0, :] = 0.0
    return alpha


class SessaMixer(nn.Module):
    def __init__(
        self,
        D: int,
        n_heads: int = 1,
        n_kv_heads: int | None = None,
        max_len: int | None = None,
        use_flash: bool = False,
        use_forward_rope: bool = True,
        gamma_max: float = 0.999,
    ):
        super().__init__()
        self.D = D
        self.max_len = max_len

        self.flash_enabled = bool(use_flash and _FLASH_ATTN_AVAILABLE)
        self.use_forward_rope = bool(use_forward_rope)
        
        if not (0.0 < float(gamma_max) < 1.0):
            raise ValueError("gamma_max must be in (0,1).")
        self.gamma_max = float(gamma_max)

        self.n_heads = int(n_heads)
        if D % self.n_heads != 0:
            raise ValueError("D must be divisible by n_heads.")
        self.head_dim = D // self.n_heads

        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else self.n_heads
        if self.n_kv_heads <= 0 or self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads must be in [1, n_heads].")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")

        if self.use_forward_rope and (self.head_dim % 2 != 0):
            raise ValueError("Head dim must be even to apply RoPE over even/odd dimension pairs.")

        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(D, D, bias=True)
        self.k_proj = nn.Linear(D, kv_dim, bias=True)
        self.v_proj = nn.Linear(D, kv_dim, bias=True)

        self.proj_fb = nn.Linear(D, 2 * D, bias=True)    

        self.out_proj = nn.Identity() if self.n_heads == 1 else nn.Linear(D, D, bias=False)

        self.gamma_bias = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gamma_ctx = nn.Linear(D, 1, bias=False)

        if self.use_forward_rope:
            half = self.head_dim // 2
            inv_freq = 10000.0 ** (-2 * torch.arange(half, dtype=torch.float32) / self.head_dim)
            self.register_buffer("rope_inv_freq", inv_freq)
        else:
            self.rope_inv_freq = None

        if max_len is not None:
            base = torch.ones(max_len, max_len, dtype=torch.bool)
            self.register_buffer("causal_mask_f_full", torch.triu(base, diagonal=1))
            self.register_buffer("causal_mask_fb_full", torch.triu(base, diagonal=0))
        else:
            self.causal_mask_f_full = None
            self.causal_mask_fb_full = None
            
    def _get_masks(self, T: int, device):
        if self.causal_mask_f_full is not None:
            return (
                self.causal_mask_f_full[:T, :T].to(device),
                self.causal_mask_fb_full[:T, :T].to(device),
            )
        base = torch.ones(T, T, dtype=torch.bool, device=device)
        causal_mask_f = torch.triu(base, diagonal=1)
        causal_mask_fb = torch.triu(base, diagonal=0)
        return causal_mask_f, causal_mask_fb

    @staticmethod
    def _apply_rope(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # x: (B,T,H,head_dim), theta: (B,T,half) where half=head_dim//2
        B, T, H, d = x.shape
        half = d // 2
        assert theta.shape == (B, T, half)

        cos = torch.cos(theta).unsqueeze(2)  # (B,T,1,half)
        sin = torch.sin(theta).unsqueeze(2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out
    
    @staticmethod
    def _pick_flash_dtype(device: torch.device) -> torch.dtype | None:
        if device.type != "cuda":
            return None
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    @staticmethod
    def _solve_feedback_system(L: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        if L.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128) and \
           f.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            return torch.linalg.solve_triangular(
                L, f, upper=False, left=True, unitriangular=True
            )
        s = torch.linalg.solve_triangular(
            L.float(), f.float(), upper=False, left=True, unitriangular=True
        )
        return s.to(f.dtype)

    def _forward_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: (B,T,Hq,d), k,v: (B,T,Hkv,d)
        B, T, Hq, d = q.shape
        Hkv = k.shape[2]
        assert k.shape == (B, T, Hkv, d)
        assert v.shape == (B, T, Hkv, d)
        assert Hq % Hkv == 0
        g = Hq // Hkv
        sigma_k = d ** -0.5

        can_flash = self.flash_enabled and q.is_cuda and d <= 256

        if can_flash:
            try:
                orig_dtype = q.dtype
                flash_dtype = orig_dtype
                if orig_dtype not in (torch.float16, torch.bfloat16):
                    picked = self._pick_flash_dtype(q.device)
                    if picked is None:
                        raise RuntimeError("Flash attention requires CUDA.")
                    flash_dtype = picked

                q_in = q
                k_in = k
                v_in = v
                if flash_dtype != orig_dtype:
                    q_in = q_in.to(flash_dtype)
                    k_in = k_in.to(flash_dtype)
                    v_in = v_in.to(flash_dtype)

                out = flash_attn_func(
                    q_in.contiguous(),
                    k_in.contiguous(),
                    v_in.contiguous(),
                    dropout_p=0.0,
                    softmax_scale=sigma_k,
                    causal=True,
                )
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)
                return out  # (B,T,Hq,d)
            except Exception:
                pass

        device = q.device
        causal_mask_f, _ = self._get_masks(T, device)  # (T,T)
        mask = causal_mask_f.view(1, 1, 1, T, T)  # (1,1,1,T,T)

        qg = q.view(B, T, Hkv, g, d)  # (B,T,Hkv,g,d)
        logits = torch.einsum("bthgd,bshd->bhgts", qg, k) * sigma_k  # (B,Hkv,g,T,T)
        logits = logits.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhgts,bshd->bthgd", alpha, v)  # (B,T,Hkv,g,d)
        return out.reshape(B, T, Hq, d)

    def forward(self, bar_a: torch.Tensor) -> torch.Tensor:
        B, T, D = bar_a.shape
        if D != self.D:
            raise ValueError(f"Expected last dim D={self.D}, got {D}.")

        device = bar_a.device
        dtype = bar_a.dtype

        _, causal_mask_fb = self._get_masks(T, device)

        q_f = self.q_proj(bar_a)  # (B,T,D) -> (B,T,Hq,d)
        k_f = self.k_proj(bar_a)  # (B,T,Hkv*d)
        v_f = self.v_proj(bar_a)  # (B,T,Hkv*d)

        proj_fb = self.proj_fb(bar_a)
        q_b, k_b = torch.split(proj_fb, [D] * 2, dim=-1)

        q_f = q_f.view(B, T, self.n_heads, self.head_dim)
        k_f = k_f.view(B, T, self.n_kv_heads, self.head_dim)
        v_f = v_f.view(B, T, self.n_kv_heads, self.head_dim)

        if self.use_forward_rope:
            pos = torch.arange(T, device=device, dtype=torch.float32)
            angles = pos[:, None] * self.rope_inv_freq[None, :]
            theta = angles.unsqueeze(0).expand(B, T, -1).to(dtype)
            q_f_rope = self._apply_rope(q_f, theta)
            k_f_rope = self._apply_rope(k_f, theta)
        else:
            q_f_rope, k_f_rope = q_f, k_f

        f = self._forward_attention(q_f_rope, k_f_rope, v_f)  # (B,T,Hq,d)
        f = f.reshape(B, T, D)  # (B,T,D)           
        f = self.out_proj(f)

        sigma_fb = D ** -0.5
        logits_fb = torch.einsum("btd,bsd->bts", q_b, k_b) * sigma_fb      # (B,T,T)
        alpha_fb = causal_softmax(logits_fb, causal_mask_fb)

        gamma_raw = self.gamma_bias.view(1, 1, 1) + self.gamma_ctx(bar_a)  # (B,T,1)
        gamma = self.gamma_max * torch.tanh(gamma_raw)  # (B,T,1)
        B_fb = gamma * alpha_fb  # (B,T,T)

        L = -B_fb
        s = self._solve_feedback_system(L, f)  # (B,T,D)
        return s
    
        
    def init_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_len: int | None = None,
    ) -> dict:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        L = int(max_len if max_len is not None else (self.max_len if self.max_len is not None else 0))
        if L <= 0:
            raise ValueError("Need max_len (arg or self.max_len) to preallocate KV cache.")

        B = int(batch_size)
        Hkv = self.n_kv_heads
        d = self.head_dim
        D = self.D

        cache = {
            "t": 0,
            "k_f": torch.empty(B, L, Hkv, d, device=device, dtype=dtype),
            "v_f": torch.empty(B, L, Hkv, d, device=device, dtype=dtype),
            "k_b": torch.empty(B, L, D, device=device, dtype=dtype),
            "s":   torch.empty(B, L, D, device=device, dtype=dtype),
        }
        return cache


    def prefill(self, bar_a: torch.Tensor, cache: dict) -> dict:
        B, T, D = bar_a.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got {D}.")
        if cache["t"] != 0:
            raise ValueError("prefill expects empty cache (t==0).")
        if T > cache["k_f"].shape[1]:
            raise ValueError("prefill length exceeds cache max_len.")

        device = bar_a.device
        dtype = bar_a.dtype

        if T == 0:
            cache["t"] = 0
            return cache

        q_f = self.q_proj(bar_a).view(B, T, self.n_heads, self.head_dim)              # (B,T,Hq,d)
        k_f = self.k_proj(bar_a).view(B, T, self.n_kv_heads, self.head_dim)           # (B,T,Hkv,d)
        v_f = self.v_proj(bar_a).view(B, T, self.n_kv_heads, self.head_dim)           # (B,T,Hkv,d)

        if self.use_forward_rope:
            pos = torch.arange(T, device=device, dtype=torch.float32)
            angles = pos[:, None] * self.rope_inv_freq[None, :]
            theta = angles.unsqueeze(0).expand(B, T, -1).to(dtype)
            q_f_rope = self._apply_rope(q_f, theta)
            k_f_rope = self._apply_rope(k_f, theta)
        else:
            q_f_rope, k_f_rope = q_f, k_f

        cache["k_f"][:, :T] = k_f_rope
        cache["v_f"][:, :T] = v_f

        f = self._forward_attention(q_f_rope, k_f_rope, v_f)  # (B,T,H,head_dim)
        f = f.reshape(B, T, D)                                # (B,T,D)
        f = self.out_proj(f)                                  # (B,T,D)

        # --- (I - B_fb) s = f ---
        proj_fb = self.proj_fb(bar_a)
        q_b, k_b = torch.split(proj_fb, [D] * 2, dim=-1)
        cache["k_b"][:, :T] = k_b

        _, causal_mask_fb = self._get_masks(T, device)
        sigma_fb = D ** -0.5
        logits_fb = torch.einsum("btd,bsd->bts", q_b, k_b) * sigma_fb      # (B,T,T)
        alpha_fb = causal_softmax(logits_fb, causal_mask_fb)              # (B,T,T)

        gamma_raw = self.gamma_bias.view(1, 1, 1) + self.gamma_ctx(bar_a) # (B,T,1)
        gamma = self.gamma_max * torch.tanh(gamma_raw)                    # (B,T,1)
        B_fb = gamma * alpha_fb                                           # (B,T,T)

        Ltri = -B_fb
        s = self._solve_feedback_system(Ltri, f)  # (B,T,D)
        cache["s"][:, :T] = s

        cache["t"] = T
        return cache


    def decode_step(self, bar_a_t: torch.Tensor, cache: dict, use_flash_decode: bool = False) -> tuple[torch.Tensor, dict]:
        if bar_a_t.dim() == 3:
            if bar_a_t.shape[1] != 1:
                raise ValueError("decode_step expects (B,1,D) if 3D input.")
            bar_a_t = bar_a_t[:, 0, :]
        if bar_a_t.dim() != 2:
            raise ValueError("decode_step expects bar_a_t shape (B,D) or (B,1,D).")

        B, D = bar_a_t.shape
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got {D}.")

        t = int(cache["t"])
        L = cache["k_f"].shape[1]
        if t >= L:
            raise ValueError("Cache is full (t >= max_len).")

        use_flash_decode = bool(use_flash_decode)

        device = bar_a_t.device
        dtype = bar_a_t.dtype

        q_f_t = self.q_proj(bar_a_t).view(B, 1, self.n_heads, self.head_dim)          # (B,1,Hq,d)
        k_f_t = self.k_proj(bar_a_t).view(B, 1, self.n_kv_heads, self.head_dim)       # (B,1,Hkv,d)
        v_f_t = self.v_proj(bar_a_t).view(B, 1, self.n_kv_heads, self.head_dim)       # (B,1,Hkv,d)

        if self.use_forward_rope:
            pos = torch.tensor([t], device=device, dtype=torch.float32)  # (1,)
            angles = pos[:, None] * self.rope_inv_freq[None, :]          # (1, half)
            theta = angles.unsqueeze(0).expand(B, 1, -1).to(dtype)       # (B,1,half)
            q_f_t = self._apply_rope(q_f_t, theta)
            k_f_t = self._apply_rope(k_f_t, theta)

        proj_fb_t = self.proj_fb(bar_a_t)  # (B, 2D)
        q_b_t, k_b_t = torch.split(proj_fb_t, [D] * 2, dim=-1)  # (B,D), (B,D)

        cache["k_b"][:, t] = k_b_t       # (B,D)

        sigma_k = (self.head_dim) ** -0.5

        use_flash_kvcache = (
            self.flash_enabled
            and _FLASH_KVCACHE_AVAILABLE
            and (flash_attn_with_kvcache is not None)
            and use_flash_decode              
            and (self.n_heads > 1)           
            and q_f_t.is_cuda
            and cache["k_f"].is_cuda
            and cache["v_f"].is_cuda
            and self.head_dim <= 256
            and cache["k_f"].dtype in (torch.float16, torch.bfloat16)
            and cache["v_f"].dtype == cache["k_f"].dtype
            and q_f_t.dtype == cache["k_f"].dtype
        )

        if use_flash_kvcache:
            out = flash_attn_with_kvcache(
                q_f_t.contiguous(),                # (B,1,Hq,d)
                cache["k_f"],                      # (B,L,Hkv,d)
                cache["v_f"],                      # (B,L,Hkv,d)
                k=k_f_t.contiguous(),              # (B,1,Hkv,d)
                v=v_f_t.contiguous(),              # (B,1,Hkv,d)
                cache_seqlens=t,                   
                softmax_scale=sigma_k,
                causal=True,
            )
            if isinstance(out, tuple):
                out = out[0]
            out = out.contiguous()
        else:
            cache["k_f"][:, t:t+1] = k_f_t   # (B,1,Hkv,d)
            cache["v_f"][:, t:t+1] = v_f_t   # (B,1,Hkv,d)

            Hq = self.n_heads
            Hkv = self.n_kv_heads
            if Hq % Hkv != 0:
                raise RuntimeError("Invalid GQA config: n_heads must be divisible by n_kv_heads.")
            g = Hq // Hkv

            K = cache["k_f"][:, :t+1]    # (B, t+1, Hkv, d)
            V = cache["v_f"][:, :t+1]    # (B, t+1, Hkv, d)

            qg = q_f_t.view(B, 1, Hkv, g, self.head_dim)  # (B,1,Hkv,g,d)
            logits = torch.einsum("bthgd,bshd->bhgts", qg, K) * sigma_k  # (B,Hkv,g,1,t+1)
            alpha = torch.softmax(logits, dim=-1)
            out = torch.einsum("bhgts,bshd->bthgd", alpha, V)  # (B,1,Hkv,g,d)
            out = out.reshape(B, 1, Hq, self.head_dim)         # (B,1,Hq,d)
            out = out.contiguous()

        f_t = out.reshape(B, D)          # (B,D)
        f_t = self.out_proj(f_t)         # (B,D)

        gamma_raw = self.gamma_bias.view(1, 1) + self.gamma_ctx(bar_a_t)  # (B,1)
        gamma = self.gamma_max * torch.tanh(gamma_raw)                    # (B,1)

        if t == 0:
            s_t = f_t
        else:
            sigma_fb = D ** -0.5  
            Kb = cache["k_b"][:, :t]     # (B,t,D)
            logits_fb = torch.einsum("bd,bsd->bs", q_b_t, Kb) * sigma_fb
            alpha_fb = torch.softmax(logits_fb, dim=-1)  # (B,t)

            S_prev = cache["s"][:, :t]   # (B,t,D)
            ctx = torch.einsum("bs,bsd->bd", alpha_fb, S_prev)  # (B,D)
            s_t = f_t + gamma * ctx      # (B,D) + (B,1)*(B,D)

        cache["s"][:, t] = s_t
        cache["t"] = t + 1
        return s_t, cache