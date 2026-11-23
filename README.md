# Building a Transformer Language Model from Scratch

**Author:** Yunwei Mo · central_planner@sjtu.edu.cn · [GitHub](https://github.com/CPP-quant)

This repository is my implementation of
[Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
(Spring 2025), Assignment 1. The task is to rebuild the core of a GPT-2-style
language model — tokenizer included — from raw tensor operations. Attention,
normalization, and the cross-entropy loss are all written out directly rather than
assembled from `torch.nn.functional`, and the tokenizer is trained from bytes
without any external BPE library.

## What the assignment covers

Assignment 1 takes a blank Python package and asks for the full stack behind a
decoder-only Transformer:

- **A byte-level BPE tokenizer** trained from scratch — pre-tokenization,
  pair counting, greedy merging, and the encode/decode paths — with no external
  tokenizer library.
- **The model itself**: embedding lookup, RMSNorm, rotary position embeddings,
  SwiGLU feed-forward layers, multi-head self-attention, pre-norm residual
  blocks, and the language-model head.
- **The training stack**: AdamW with decoupled weight decay, a warmup-then-cosine
  learning-rate schedule, global-norm gradient clipping, memory-mapped batch
  loading, and checkpoint serialization.
- **The numerics**: stable softmax, log-sum-exp cross-entropy, and initialization
  schemes that keep activations in range without normalization tricks.

## Architecture

| Component | File | Note |
|---|---|---|
| `Linear` | `train_linear.py` | truncated-normal init (σ = 0.02), no bias |
| `Embedding` | `train_embedding.py` | lookup table over the vocabulary |
| `RMSNorm` | `train_rmsnorm.py` | epsilon inside the square root, no mean-centering |
| `SwiGLU` | `train_swiglu.py` | gated feed-forward, three projections |
| `RoPE` | `train_rope.py` | cached cos/sin, rotation on even/odd coordinate pairs |
| `softmax` | `train_softmax.py` | max-subtraction for numerical stability |
| `scaled_dot_product_attention` | `train_scaled_dot_product_attention.py` | explicit QKᵀ/√d_k, causal mask, V |
| `MultiHeadSelfAttention` | `train_multihead_self_attention*.py` | manual head split and concat, RoPE on Q and K |
| `TransformerBlock` | `train_transformer_block.py` | pre-norm residual connections |
| `TransformerLM` | `train_transformer_lm.py` | embedding → N blocks → RMSNorm → LM head |
| `cross_entropy` | `train_cross_entropy.py` | hand-written log-sum-exp |
| BPE trainer | `train_bpe.py` | GPT-2 pre-tokenization, frequency aggregation, pair merging |
| Tokenizer | `train_tokenizer.py` | vocab/merge serialization, encode, decode, special tokens |

## What I took away from it

**Tokenization hides more subtlety than the model does.**
BPE looks like a textbook greedy algorithm until tie-breaking starts to matter.
The reference GPT-2 tokenizer trains on a `bytes_to_unicode` alphabet, but that
mapping does not preserve byte order — byte 32 becomes `Ġ` (U+0120) while byte 33
becomes `!` (U+0021). Comparing merged pairs on mapped characters can therefore
break ties in a different order than the reference, and the merge list diverges
after a few thousand merges. Training directly on raw bytes keeps the merge order
consistent. Getting the encoder to reproduce `tiktoken` token-for-token was the
point where the algorithm actually clicked.

**Numerical stability is a design decision, not boilerplate.**
Writing `exp(x − max(x)) / Σ exp(x − max(x))` by hand makes it obvious why the
subtraction exists: without it, logits in the tens overflow fp32 immediately. The
same reasoning carries into cross-entropy, where `log(Σ exp(shifted)) −
shifted[target]` never materializes a probability vector at all. RMSNorm puts
epsilon inside the square root for the same reason — a zero-variance input must
not turn into a division by zero.

**Positional encoding shapes what the model can even learn.**
Implementing RoPE as an explicit 2×2 rotation over coordinate pairs, rather than
adding a position vector, made the relative-position property concrete: the dot
product of a rotated query and key depends only on their offset. I cache cos/sin
for `max_seq_len` positions and index them with `token_positions` instead of
slicing `[:seq_len]`, which keeps the layer correct for non-contiguous indices.

**Constraints change the algorithm, not just the runtime.**
The naive version of BPE recounts every adjacent pair after every merge, which is
hopeless on a multi-megabyte corpus. Aggregating the corpus into a histogram of
pre-tokenized words first means a word seen ten thousand times enters the merge
loop once with weight 10⁴, and pair counts update incrementally instead of being
recomputed. Working without a GPU made this kind of reasoning unavoidable: the
only way through the corpus was to stop doing redundant work.

## Environment

Developed on Windows 11 with CPU only. Two of the assignment's memory-usage checks
require Linux `resource` limits and are skipped there; a small shim
(`resource.py`) keeps the import working on Windows.

## Quickstart

The environment is managed with [uv](https://github.com/astral-sh/uv).

```sh
uv sync              # build the environment from uv.lock
uv run pytest        # run the reference test suite
```

Implementations live in `cs336_basics/`; the test suite reaches them through
`tests/adapters.py`.
