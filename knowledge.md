# Knowledge Base

## Текущая гипотеза
exp_016: 12 layers, 10min, lr=0.04 — more capacity. Layer sweep at 10-min to find optimal depth.

## Лучший результат
- **exp_014 (9 layers, lr=0.04, H100/10min): 1.3583 bpb** (990 steps, 606ms/step, 12.4MB compressed)
- Target baseline: **1.2244 bpb** (8xH100, 10min) — gap: 0.134

## Current train_gpt.py defaults
- 3 layers, model_dim=512, num_heads=8, num_kv_heads=4 (GQA)
- matrix_lr=0.10, vocab_size=1024, seq_len=1024, tied_embeddings=True
- MLP: relu^2, 2x expansion
- num_loop_iters=1 (looped layers available but off)

## Что работает (H100/10min)
- **9 layers, 10min**: 1.3583 bpb. Much better than any 2-min run. Capacity matters with enough steps.
- **LR is NOT sensitive at 10min**: lr=0.04→1.358, lr=0.10→1.360. Don't waste time on LR at 10min.

## Что работает (H100/2min — screening only, may not transfer)
- **Higher matrix_lr**: 0.04→1.58, 0.08→1.55, 0.12→1.549 (optimal ~0.10 for 2-min)
- **3-4 layers**: sweet spot for 2-min only

## Что не работает (2-min, needs retesting at 10-min)
- **SwiGLU (A40 only)**: +0.067 bpb. RETEST on H100/10min — enough steps to amortize slower step time.
- **Wider dim (768, 2min)**: +0.15 bpb. RETEST at 10min.
- **Looped layers (3×2, 2min)**: +0.12 bpb. RETEST at 10min — compute cost may pay off.

## Что точно не работает
- **2 layers (H100/2min)**: insufficient capacity even for 2-min
- **Smaller batch (262144)**: less informative steps regardless of training time
- **Higher LR at 10min**: basically no effect (exp_014 vs exp_015)

## Compression awareness
- 9L/lr=0.04: 12.4MB (3.6MB headroom)
- 9L/lr=0.10: 14.6MB (dangerously close!) — higher LR → larger weights → worse compression
- Headroom allows ~2 more layers or modest width increase at lr=0.04

## Тупики
(пока пусто)

## Паттерны и инсайты
- **Regime shift**: 2-min and 10-min are DIFFERENT optimization problems
  - 2-min: speed dominates, small models win, LR matters
  - 10-min: capacity dominates, larger models win, LR barely matters
- **10-min gives 990 steps** at 9L (606ms/step). Good convergence.
- **Final submit is 8xH100/10min** — 8x more compute than our 1xH100 runs. Bigger models will work even better there.
- **Compressed size is binding constraint at 10min**, not speed.

## Layer trend (H100/10min)
- 9L→1.358 (need: 6L, 12L, 4L to find optimal)

## Priority queue
1. **Layer sweep at 10min**: 12L (running), then 6L, 4L
2. **Architecture at optimal depth**: SwiGLU, wider dim, GQA tuning
3. **Training improvements**: warmup_steps, warmdown tuning
4. **Compression**: if over 16MB, try int4 QAT or reduce dim
