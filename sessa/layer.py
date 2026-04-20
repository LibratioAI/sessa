import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixer import SessaMixer


class SessaLayer(nn.Module):
    def __init__(
        self,
        D: int,
        n_heads: int = 1,
        n_kv_heads: int | None = None,
        max_len: int | None = None,
        ln_eps: float = 1e-5,
        use_flash: bool = False,
        use_forward_rope: bool = True,
        gamma_max: float = 0.999,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(D, eps=ln_eps)
        self.W_in = nn.Linear(D, 2 * D, bias=True)
        self.mixer = SessaMixer(
            D=D,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_len=max_len,
            use_flash=use_flash,
            use_forward_rope=use_forward_rope,
            gamma_max=gamma_max,
        )
        self.W_out = nn.Linear(D, D, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xLN = self.ln(x)
        a, g = self.W_in(xLN).chunk(2, dim=-1)
        bar_a = F.gelu(a)
        s = self.mixer(bar_a)
        y = x + self.W_out(s * g)
        return y
        

    def init_cache(self, batch_size: int, device=None, dtype=None, max_len=None) -> dict:
        return self.mixer.init_cache(batch_size, device=device, dtype=dtype, max_len=max_len)

    def prefill(self, x: torch.Tensor, cache: dict) -> tuple[torch.Tensor, dict]:
        xLN = self.ln(x)
        a, g = self.W_in(xLN).chunk(2, dim=-1)
        bar_a = F.gelu(a)

        cache = self.mixer.prefill(bar_a, cache)   

        s = cache["s"][:, :bar_a.shape[1]]
        y = x + self.W_out(s * g)
        return y, cache

    def decode_step(self, x_t: torch.Tensor, cache: dict, use_flash_decode: bool = False) -> tuple[torch.Tensor, dict]:
        was_3d = (x_t.dim() == 3)
        if was_3d:
            if x_t.shape[1] != 1:
                raise ValueError("decode_step expects x_t shape (B,D) or (B,1,D).")
            x_t_2d = x_t[:, 0, :]
        elif x_t.dim() == 2:
            x_t_2d = x_t
        else:
            raise ValueError("decode_step expects x_t shape (B,D) or (B,1,D).")
        xLN_t = self.ln(x_t_2d)
        a_t, g_t = self.W_in(xLN_t).chunk(2, dim=-1)
        bar_a_t = F.gelu(a_t)
        s_t, cache = self.mixer.decode_step(bar_a_t, cache, use_flash_decode=use_flash_decode)
        y_t_2d = x_t_2d + self.W_out(s_t * g_t)

        if was_3d:
            return y_t_2d.unsqueeze(1), cache  # (B,1,D)
        return y_t_2d, cache