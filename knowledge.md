# Knowledge Base

## Текущая гипотеза
exp_014: Run 9 layers (original baseline) on H100 with 10 min (600s) to see how much capacity matters with enough training time. Target: beat 1.2244 baseline.

## Лучший результат
- **exp_012 (3 layers, lr=0.12, H100/2min): 1.5489 bpb** (444 steps, 270ms/step, 4.0MB)
- Target baseline: **1.2244 bpb** (8xH100, 10min)

## Current train_gpt.py defaults
- 3 layers, model_dim=512, num_heads=8, num_kv_heads=4 (GQA)
- matrix_lr=0.10, vocab_size=1024, seq_len=1024, tied_embeddings=True
- MLP: relu^2, 2x expansion
- num_loop_iters=1 (looped layers available but off)

## Что работает (H100/2min)
- **Higher matrix_lr**: 0.04→1.58, 0.08→1.55, 0.12→1.549 (diminishing returns, optimal ~0.10)
- **3-4 layers**: sweet spot for 2-min H100 runs (3L: 1.58 at lr=0.04, 4L: 1.59)
- **Fewer layers on A40**: -0.436 bpb (9→6 layers). A40-specific due to slow steps.

## Что не работает
- **SwiGLU (exp_002, A40 only)**: +0.067 bpb. Needs retesting on H100.
- **Wider dim (768) with 4 layers**: +0.15 bpb. Too slow (423ms vs 305ms/step).
- **Wider dim (640) with 3 layers**: +0.016 bpb. Marginal slowdown not worth it.
- **2 layers**: -0.035 bpb vs 3 layers. Insufficient capacity.
- **Looped layers (3×2)**: +0.12 bpb. Doubles compute without enough benefit.
- **Smaller batch (262144)**: +0.03 bpb. Less informative steps.

## Тупики
(пока пусто)

## Паттерны и инсайты
- **H100 2-min**: ~270-406ms/step depending on model size. 3 layers optimal for 2-min screening.
- **LR trend**: matrix_lr 0.04→1.58, 0.08→1.55, 0.12→1.549. Optimal near 0.10.
- **Layer trend (H100/2min)**: 6→1.70, 4→1.59, 3→1.58, 2→1.61. U-shape, optimal at 3.
- **Key insight**: 2-min results (best ~1.55) are far from target (1.22). Need 10-min runs and larger models to close the gap.
- **Size budget**: 3.5-5.8MB compressed << 16MB limit. Huge room for more params.

## Next priorities
1. Run baseline (6-9 layers) with 10 min — see real convergence
2. Try SwiGLU on H100 with enough training time
3. Try looped layers with 10 min (compute cost may pay off)
4. Explore 12+ layers with 10 min
5. LR schedule: cosine decay, different warmup
