# Stanford CS336: Language Modeling from Scratch

**Assignment 1 — Basics, written without a GPU**

A **self-study** implementation of Stanford CS336, built against the official staff
test suite. Thanks to the CS336 instructors for open-sourcing such a first-principles
curriculum.

Yunwei Mo is a first-year undergraduate in the School of Computer Science at Shanghai
Jiao Tong University, majoring in Information Security (Class of 2030). Not enrolled
for credit; no Gradescope or leaderboard submissions. Implementations live in this
repository, while datasets, checkpoints and the assignment PDF stay local.

## Progress

| Assignment | Topic | Progress |
| :--- | :--- | :---: |
| Assignment 1 | Basics — tokenizer, Transformer, optimizer, training | 100% |
| Assignment 2 | Systems — FlashAttention-2, distributed training | 0% |
| Assignment 3 | Scaling — scaling laws | 0% |
| Assignment 4 | Data — filtering and deduplication | 0% |
| Assignment 5 | Alignment — SFT, RL, DPO | 0% |

## Assignment 1

About 1,400 lines in `cs336_basics/`. Attention, normalization and the cross-entropy
loss are written from raw tensor operations; the tokenizer is trained from bytes with
no external BPE library.

- **Tokenizer** — byte-level BPE from scratch. `encode` reproduces OpenAI's `tiktoken`
  token-for-token on a TinyStories sample, which means the merge *order* agrees, not
  just the final vocabulary.
- **Model** — `Linear`, `Embedding`, `RMSNorm`, `SiLU` / `SwiGLU`, `RoPE`, `softmax`,
  scaled dot-product attention, multi-head attention, pre-norm Transformer block, and
  the LM head.
- **Training** — `AdamW` with bias correction and decoupled weight decay, linear warmup
  plus cosine schedule, global-norm gradient clipping, memory-mapped batch loading, and
  checkpointing.

All 46 runnable tests pass. Two further tests need Unix `resource` limits and are
skipped on Windows.

## Log

- **2026-11-15 → 11-17** — `train_bpe.py`, fitted around coursework across three
  evenings. One trap worth recording: the corpus has to be read in text mode, since
  Python normalizes CRLF while the reference merges were built on normalized text —
  reading it as binary trains a token the official vocabulary does not contain.
- **2026-11-18** — BPE passes in the morning. Tokenizer written next, and `encode`
  matches `tiktoken` token-for-token by mid-afternoon — that checks the merge *order*,
  not just the final vocabulary. Then turned to the model: `Linear`, `Embedding`,
  `RMSNorm`.
- **2026-11-19** — `SwiGLU`; `RoPE` at 11:31 after deriving θ_{i,k} = i / Θ^{2k/d_k}
  by hand. What took the thinking was not the formula but seeing why a fixed rotation
  encodes *relative* offset at all — the dot product of a rotated query and key
  collapses to a function of their distance. `softmax` in the evening, then scaled
  dot-product attention and multi-head attention five minutes apart.
- **2026-11-20** — RoPE fused into attention; the pre-norm block built so the residual
  shortcut stays an untouched identity path; full `TransformerLM` running by midday.
  Cross-entropy in the evening — hand-written log-sum-exp, no `F.cross_entropy`.
- **2026-11-23** — Went back and cleaned up `softmax` and `cross_entropy` first, then
  the optimizer: `AdamW` with lazy state init, bias correction and decoupled weight
  decay; warmup + cosine schedule; global-norm gradient clipping.
- **2026-11-24** — Data loading over a memory-mapped token array, checkpoint
  round-trip, and `SiLU` pulled into its own module so `SwiGLU` stops calling
  `F.silu`. Training stack complete; all runnable tests green.

## Environment

Windows 11, CPU only — no GPU at any point. `resource.py` is a small shim that keeps
`import resource` in the test suite importable on Windows.

## Quickstart

The environment is managed with [uv](https://github.com/astral-sh/uv).

```sh
uv sync              # build the environment from uv.lock
uv run pytest        # run the reference test suite
```

Implementations live in `cs336_basics/`; the test suite reaches them through
`tests/adapters.py`.
