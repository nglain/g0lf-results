# Knowledge Base

## Текущая гипотеза
Sliding window eval CONFIRMED working (-0.012 bpb at stride=2048). Next: try NTK RoPE eval-time scaling.

## Лучший результат
- **exp_025 (9L, seq4096, H100/10min): 1.3415 bpb with sliding eval** (stride=2048)
  - Standard roundtrip: 1.3537, sliding: 1.3415 = -0.012 bpb FREE
- Target baseline: **1.2244 bpb** (8xH100, 10min) — gap: 0.117 (with slide eval)
- NO val-only training (forbidden)

## Current train_gpt.py state
- Defaults: 3 layers, dim=512, lr=0.10 (for quick screening)
- Best 10min config: 9L/seq4096/lr=0.04 → 1.3525 (standard eval), 1.3415 (sliding)
- Features: looped layers (off), sliding window eval (SLIDE_STRIDE), forward_logits, int6 quant (INT6_LAYERS)

## Что работает (H100/10min)
- **Sliding window eval**: -0.012 bpb FREE at stride=2048. With stride=64 on 8xH100, expect -0.03-0.05. CONFIRMED.
- **9L seq4096**: 1.3525 (standard). Best architecture for 1xH100.
- **6L seq1024**: 1.3573. Same quality but only 9.5MB. Good if size constrained.
- All seq_len variants (1024, 2048, 4096) give ~1.352-1.358 on 1xH100. Minimal effect.

## Что не работает (H100/10min, 1xH100)
- **Competition Muon tuning**: lr=0.02/mom=0.99 → 1.414 (WORSE). Needs 9500 steps.
- **Moderate Muon tuning**: mom=0.97/warmdown=2000 → 1.353 (no change).
- **MLP 3x**: 1.356 (worse). More params = slower = fewer steps. 14.8MB near limit.
- **10L + int6**: 1.364 pre-quant, 1.385 post-quant. Int6 causes +0.02 roundtrip degradation.
- **12L**: 1.373 (worse). Too slow.
- **Wider dim (640, 768)**: worse at any training duration.
- **Looped layers**: worse. PR #31 confirmed.

## Compression awareness
- 9L/seq4096: 11.9MB (4.1MB headroom)
- 6L/seq1024: 9.5MB (6.5MB headroom)

## Key insight
On 1xH100 with ~890 steps, we're compute-bound. ALL architecture changes trade steps for capacity and net to ~1.35. The only free improvements are eval-time techniques:
- Sliding window eval: ✅ CONFIRMED
- NTK RoPE eval: TODO
- Any other eval-time compute trick

## NEW competition intel (2026-03-19 16:30)
- **PR #85: 92-experiment autoresearch + sliding window = pre-quant 1.2156 BPB** — BEATS BASELINE!
  - Similar approach to ours (autoresearch). They ran 92 experiments.
  - Key: they use 8xH100 with full 10min budget.
- **PR #81: depth recurrence 4×3 + SwiGLU + int6 = 1.2269 on 4×H100** — competitive!
  - First time depth recurrence works. Needs SwiGLU + int6 + enough compute.
- **PR #84: mirrored-recurrence MLX** — non-record
- 85 PRs total, 0 merged. Race for first accepted record is OPEN.
- **Doc-isolated eval (PR #77 ablation)**: reset state between documents = -0.011 FREE. Easy to implement.

## Priority queue (updated)
1. ✅ Sliding window eval — DONE, confirmed -0.012 bpb
2. NTK RoPE eval-time scaling (PR #60, free improvement)
3. Doc-isolated eval (reset between docs, free -0.011 from PR #77 ablation)
4. Try warmdown_iters tuning (shorter warmdown since we stop early)
5. Optimize for 8xH100 final: set defaults to competition-proven params
