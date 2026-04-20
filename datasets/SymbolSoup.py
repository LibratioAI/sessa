import argparse
import csv
import os
import random
from math import ceil, floor
from typing import Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm



N_LEFT_STYLES = 4
N_RIGHT_STYLES = 5

CLASSES = [f"CLASS_{i}" for i in range(N_LEFT_STYLES * N_RIGHT_STYLES)]

VOCAB = list("abcdefghijklmnop")
NOISE_VOCAB = VOCAB

BIGRAM_STRENGTH = 3.0
MOTIF_INSERT_PROB = 0.3
NOISE_FLIP_PROB = 0.03

BOS = "<bos>"
SEP1 = "<sep1>"
SEP2 = "<sep2>"
SEP3 = "<sep>"
LABEL_TOKEN = "<label>"
EOS = "<eos>"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = [BOS, SEP1, SEP2, SEP3, LABEL_TOKEN, EOS, MASK_TOKEN]




def tokenize_with_specials(text: str) -> List[str]:
    """Special tokens are single tokens; everything else is split into single characters."""
    tokens: List[str] = []
    i = 0
    n = len(text)
    ordered_specials = sorted(SPECIAL_TOKENS, key=len, reverse=True)

    while i < n:
        matched = False
        for s in ordered_specials:
            if text.startswith(s, i):
                tokens.append(s)
                i += len(s)
                matched = True
                break
        if matched:
            continue
        tokens.append(text[i])
        i += 1

    return tokens




def build_motifs(vocab: List[str]) -> List[str]:
    """All contiguous substrings of lengths 3, 4, 5 over the vocab."""
    motifs: List[str] = []
    n = len(vocab)

    for L in [3, 4, 5]:
        if L > n:
            continue
        for start in range(n - L + 1):
            motifs.append("".join(vocab[start:start + L]))

    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for m in motifs:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


MOTIFS = build_motifs(VOCAB)


def build_style_configs(
    num_styles: int,
    rng_seed: int,
) -> Tuple[
    Dict[int, Dict[str, float]],
    Dict[int, Set[Tuple[str, str]]],
    Dict[int, Dict[str, float]],
]:
    """
    Build per-style configs:
      - unigram weights
      - preferred bigrams (prev, cur)
      - motif weights
    """
    cfg_rng = random.Random(rng_seed)

    style_unigram: Dict[int, Dict[str, float]] = {}
    style_bigram: Dict[int, Set[Tuple[str, str]]] = {}
    style_motif_weights: Dict[int, Dict[str, float]] = {}

    for s in range(num_styles):
        tokens = VOCAB.copy()
        cfg_rng.shuffle(tokens)
        primaries = tokens[:4]
        secondaries = tokens[4:8]
        others = tokens[8:]

        uni: Dict[str, float] = {}
        for t in primaries:
            uni[t] = 3.0
        for t in secondaries:
            uni[t] = 2.0
        for t in others:
            uni[t] = 1.0
        style_unigram[s] = uni

        big: Set[Tuple[str, str]] = set()
        for i in range(len(primaries)):
            a = primaries[i]
            b = primaries[(i + 1) % len(primaries)]
            big.add((a, b))
            big.add((a, a))
        for s2 in secondaries:
            big.add((s2, s2))
        for p, s2 in zip(primaries, secondaries):
            big.add((p, s2))
            big.add((s2, p))
        style_bigram[s] = big

        prim_set = set(primaries)
        sec_set = set(secondaries)
        mw: Dict[str, float] = {}
        for motif in MOTIFS:
            chars = set(motif)
            overlap_primary = len(chars & prim_set)
            overlap_secondary = len(chars & sec_set)
            base = 1.0
            w = base + 0.9 * overlap_primary + 0.5 * overlap_secondary
            mw[motif] = w
        style_motif_weights[s] = mw

    return style_unigram, style_bigram, style_motif_weights


LEFT_UNIGRAM, LEFT_BIGRAM, LEFT_MOTIF_WEIGHTS = build_style_configs(
    num_styles=N_LEFT_STYLES, rng_seed=123
)
RIGHT_UNIGRAM, RIGHT_BIGRAM, RIGHT_MOTIF_WEIGHTS = build_style_configs(
    num_styles=N_RIGHT_STYLES, rng_seed=456
)




def sample_from_dist(dist_dict: Dict[str, float]) -> str:
    """Sample from token->weight distribution (weights > 0, not necessarily normalized)."""
    total = sum(dist_dict.values())
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")

    r = random.random() * total
    s = 0.0
    last_tok: Optional[str] = None
    for tok, w in dist_dict.items():
        s += w
        last_tok = tok
        if r <= s:
            return tok
    assert last_tok is not None
    return last_tok


def build_wrapped_text(
    noise1_tokens: List[str],
    first_tokens: List[str],
    noise2_tokens: List[str],
    second_tokens: List[str],
    noise3_tokens: List[str],
) -> str:
    """
    Format:

        <bos>
        [noise0]
        <sep1>
        [stylized block 1]
        <sep2>
        [noise1]
        <sep1>
        [stylized block 2]
        <sep2>
        [noise2]
        <sep> <label>
    """
    text_noise1 = " ".join(noise1_tokens)
    text_first = " ".join(first_tokens)
    text_noise2 = " ".join(noise2_tokens)
    text_second = " ".join(second_tokens)
    text_noise3 = " ".join(noise3_tokens)

    full_text = (
        f"{BOS}\n"
        f"{text_noise1}\n"
        f"{SEP1}\n"
        f"{text_first}\n"
        f"{SEP2}\n"
        f"{text_noise2}\n"
        f"{SEP1}\n"
        f"{text_second}\n"
        f"{SEP2}\n"
        f"{text_noise3}\n"
        f"{SEP3} {LABEL_TOKEN}"
    )
    return full_text


def compute_M_bounds(min_tokens: int, max_tokens: int) -> Tuple[int, int]:
    """
    With 3 noise blocks + 2 stylized blocks, total letter count M satisfies:
        full_tokens = 2*M + 13
    So M in [ceil((min_tokens-13)/2), floor((max_tokens-13)/2)], with M >= 5.
    """
    M_min = max(5, ceil((min_tokens - 13) / 2))
    M_max = max(5, floor((max_tokens - 13) / 2))
    if M_min > M_max:
        raise ValueError(
            f"Cannot satisfy token length bounds: min_tokens={min_tokens}, max_tokens={max_tokens}. "
            f"Try widening the range."
        )
    return M_min, M_max


def apply_noise(tokens: List[str]) -> List[str]:
    """With probability NOISE_FLIP_PROB, replace a vocab symbol by a different random vocab symbol."""
    out: List[str] = []
    for t in tokens:
        if t in VOCAB and random.random() < NOISE_FLIP_PROB:
            choices = [x for x in VOCAB if x != t]
            out.append(random.choice(choices))
        else:
            out.append(t)
    return out


def sample_markov_step(
    unigram: Dict[str, float],
    bigrams: Set[Tuple[str, str]],
    prev: Optional[str],
) -> str:
    """
    If prev is not None:
        P(tok | prev) ∝ unigram[tok] * (1 + BIGRAM_STRENGTH) if (prev, tok) in bigrams
                     ∝ unigram[tok] otherwise
    """
    if prev is None:
        return sample_from_dist(unigram)

    weights: Dict[str, float] = {}
    for tok, base_p in unigram.items():
        w = base_p
        if (prev, tok) in bigrams:
            w = base_p * (1.0 + BIGRAM_STRENGTH)
        weights[tok] = w

    return sample_from_dist(weights)


def sample_stylized_sequence(length: int, style_id: int, is_left: bool) -> List[str]:
    """
    Stylized sequence: mix of Markov steps and motif insertions + small symbol noise.
    """
    if length <= 0:
        return []

    if is_left:
        unigram = LEFT_UNIGRAM[style_id]
        bigrams = LEFT_BIGRAM[style_id]
        motif_weights = LEFT_MOTIF_WEIGHTS[style_id]
    else:
        unigram = RIGHT_UNIGRAM[style_id]
        bigrams = RIGHT_BIGRAM[style_id]
        motif_weights = RIGHT_MOTIF_WEIGHTS[style_id]

    tokens: List[str] = []
    min_motif_len = min(len(m) for m in MOTIFS)

    while len(tokens) < length:
        remaining = length - len(tokens)
        can_insert_motif = remaining >= min_motif_len

        if can_insert_motif and random.random() < MOTIF_INSERT_PROB:
            candidates = {m: w for m, w in motif_weights.items() if len(m) <= remaining and w > 0}
            if candidates:
                motif = sample_from_dist(candidates)
                tokens.extend(list(motif))
                continue

        prev = tokens[-1] if tokens else None
        nxt = sample_markov_step(unigram, bigrams, prev)
        tokens.append(nxt)

    if len(tokens) > length:
        tokens = tokens[:length]

    tokens = apply_noise(tokens)
    return tokens


def sample_noise_sequence(length: int) -> List[str]:
    """Noise sequence: uniform iid over NOISE_VOCAB."""
    if length <= 0:
        return []
    return [random.choice(NOISE_VOCAB) for _ in range(length)]



def class_index_to_pair(cls_idx: int) -> Tuple[int, int]:
    """CLASS_k -> (left_style, right_style) via row-major indexing."""
    left_id = cls_idx // N_RIGHT_STYLES
    right_id = cls_idx % N_RIGHT_STYLES
    return left_id, right_id



def gen_example_for_class(class_label: str, min_tokens: int, max_tokens: int) -> Tuple[str, str]:
    """
    Generate one example for class_label:
      - two stylized blocks (one left-family, one right-family) in random order
      - three noise blocks
      - final: "<sep> <label>"
    """
    if class_label not in CLASSES:
        raise ValueError(f"Unknown class_label: {class_label}")

    cls_idx = CLASSES.index(class_label)
    left_style, right_style = class_index_to_pair(cls_idx)

    M_min, M_max = compute_M_bounds(min_tokens, max_tokens)
    M = random.randint(M_min, M_max)

    noise_min = max(3, int(0.3 * M))
    noise_max = max(noise_min, int(0.6 * M))
    Ln_total = random.randint(noise_min, noise_max)

    if Ln_total > M - 2:
        Ln_total = M - 2
    if Ln_total < 3:
        Ln_total = 3

    main = M - Ln_total
    if main < 2:
        main = 2
        Ln_total = M - main

    L_left = random.randint(1, main - 1)
    L_right = main - L_left

    Ln0 = random.randint(1, Ln_total - 2)
    Ln1 = random.randint(1, Ln_total - Ln0 - 1)
    Ln2 = Ln_total - Ln0 - Ln1

    seq_left = sample_stylized_sequence(L_left, left_style, is_left=True)
    seq_right = sample_stylized_sequence(L_right, right_style, is_left=False)

    if random.random() < 0.5:
        first_tokens = seq_left
        second_tokens = seq_right
    else:
        first_tokens = seq_right
        second_tokens = seq_left

    noise0_tokens = sample_noise_sequence(Ln0)
    noise1_tokens = sample_noise_sequence(Ln1)
    noise2_tokens = sample_noise_sequence(Ln2)

    text = build_wrapped_text(
        noise1_tokens=noise0_tokens,
        first_tokens=first_tokens,
        noise2_tokens=noise1_tokens,
        second_tokens=second_tokens,
        noise3_tokens=noise2_tokens,
    )
    length = len(tokenize_with_specials(text))

    if not (min_tokens <= length <= max_tokens):
        raise RuntimeError(
            f"Generated text length mismatch: {length}, expected in [{min_tokens}, {max_tokens}]"
        )

    return text, class_label



def generate_symbol_soup_examples_balanced(
    num_examples: int,
    min_tokens: int,
    max_tokens: int,
) -> List[Tuple[str, str]]:
    """Balanced over all CLASSES."""
    num_classes = len(CLASSES)
    if num_examples % num_classes != 0:
        raise ValueError(
            f"num_examples={num_examples} must be divisible by num_classes={num_classes}."
        )

    per_class = num_examples // num_classes
    data: List[Tuple[str, str]] = []

    labels: List[str] = []
    for y in CLASSES:
        labels.extend([y] * per_class)

    random.shuffle(labels)

    with tqdm(total=num_examples, desc="Generating SYMBOL_SOUP examples") as pbar:
        for y in labels:
            text, label = gen_example_for_class(
                class_label=y,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            data.append((text, label))
            pbar.update(1)

    return data



def _ensure_parent_dir(path: str) -> None:
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)


def write_tsv(path_prefix: str, data: List[Tuple[str, str]]) -> None:
    """Write TSV with header: Text \\t Answer."""
    path = path_prefix + ".tsv"
    _ensure_parent_dir(path)
    print(f"Writing {len(data)} examples to {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Text", "Answer"])
        writer.writerows(data)


def write_label_vocab(path_prefix: str) -> None:
    """Write label vocabulary TSV: Label \\t ClassId."""
    path = path_prefix + ".tsv"
    _ensure_parent_dir(path)
    print(f"Writing label vocab ({len(CLASSES)} labels) to {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Label", "ClassId"])
        for idx, lab in enumerate(CLASSES):
            writer.writerow([lab, idx])



def parse_args() -> argparse.Namespace:
    num_classes = len(CLASSES)
    parser = argparse.ArgumentParser(
        description=f"Generator for SYMBOL_SOUP ({num_classes} classes, mixed-order style blocks + noise)."
    )
    parser.add_argument("--task", type=str, default="symbol_soup")
    parser.add_argument("--num_train_samples", type=int, default=20000)
    parser.add_argument("--num_valid_samples", type=int, default=2000)
    parser.add_argument("--num_test_samples", type=int, default=2000)
    parser.add_argument("--min_tokens", type=int, default=400)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--output_dir", type=str, default="./data/symbol_soup_left_right")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    num_classes = len(CLASSES)

    for split_name, n in [
        ("train", args.num_train_samples),
        ("valid", args.num_valid_samples),
        ("test", args.num_test_samples),
    ]:
        if n % num_classes != 0:
            raise ValueError(
                f"{split_name} samples={n} must be divisible by num_classes={num_classes}."
            )

    total_examples = args.num_train_samples + args.num_valid_samples + args.num_test_samples

    num_classes = len(CLASSES)

    print("==================================================")
    print(f"SYMBOL_SOUP generator ({num_classes} classes, mixed-order style blocks + noise)")
    print(f" task           = {args.task}")
    print(f" seed           = {args.seed}")
    print(f" n_train        = {args.num_train_samples}")
    print(f" n_valid        = {args.num_valid_samples}")
    print(f" n_test         = {args.num_test_samples}")
    print(f" total examples = {total_examples}")
    print(f" min_tokens     = {args.min_tokens}")
    print(f" max_tokens     = {args.max_tokens}")
    print(f" output_dir     = {args.output_dir}")
    print("==================================================")

    train_examples = generate_symbol_soup_examples_balanced(
        num_examples=args.num_train_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    valid_examples = generate_symbol_soup_examples_balanced(
        num_examples=args.num_valid_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    test_examples = generate_symbol_soup_examples_balanced(
        num_examples=args.num_test_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    base = os.path.join(args.output_dir, args.task)

    write_tsv(base + "_train", train_examples)
    write_tsv(base + "_val", valid_examples)
    write_tsv(base + "_test", test_examples)
    write_label_vocab(base + "_labels")

    print("Done.")
    print(
        f"Train/val/test sizes: {len(train_examples)}/{len(valid_examples)}/{len(test_examples)} "
        f"(total {total_examples})"
    )


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# How to run
#
# 1) Install dependencies:
#    pip install -r requirements.txt
#
# 2) Generate dataset:
#    python symbol_soup_dataset.py \
#      --task symbol_soup \
#      --num_train_samples 20000 \
#      --num_valid_samples 2000 \
#      --num_test_samples 2000 \
#      --min_tokens 400 \
#      --max_tokens 600 \
#      --output_dir ./data/symbol_soup_left_right \
#      --seed 0
# -----------------------------------------------------------------------------