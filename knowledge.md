# Knowledge Base

## Текущая гипотеза
Sliding window eval CONFIRMED working (-0.012 bpb at stride=2048). Next: try NTK RoPE eval-time scaling.

## Лучший результат
- **exp_029 (9L, seq4096, 8xH100/10min): 1.2099 bpb, roundtrip 1.2156**
  - 7439 steps at 80ms/step. 15.8MB compressed (tight!)
  - BELOW 1.2244 baseline! With sliding eval expect ~1.20 or lower.
- Target baseline: **1.2244 bpb** — BEATEN by 0.009 (roundtrip)
- Competition frontier: ~1.16 BPB

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

## COMPETITION FRONTIER UPDATE (2026-03-19 18:00)
**Frontier moved to 1.160-1.162 BPB.** 89 PRs, 0 merged.

### PR #88: 1.1605 BPB (7 techniques stacked)
MTP auxiliary head (training-only, excluded from artifact) + MLP 3x + fp16 embed + int6 [-31,31] + zstd-22 + seq4096 + sliding window stride=512 + Muon(0.02/0.99/3000)

### PR #89: 1.1622 BPB
int6 STE (model trains knowing quant!) + fp16 embed + MLP 3x + NorMuon + SWA (7 checkpoints) + sliding window stride=64

### Key new techniques to implement:
- **MTP auxiliary head**: training-only, free gradient enrichment, zero eval cost
- **int6 STE**: fake-quantize during training → gap only +0.002 (vs +0.02 post-training)
- **SWA**: average 7 checkpoints during warmdown, free quality boost
- **zstd-22**: ~25% better compression than zlib
- **Doc-isolated eval**: reset state between documents = -0.011 FREE

## CRITICAL BUG FIX for 1xH100 (from PR #94)
⚠️ Default WARMDOWN_ITERS=1200 but we only get ~890-1000 steps on 1xH100.
This means LR decays from step 0! Setting WARMDOWN_ITERS=100 gave +0.0013 BPB.
**Try WARMDOWN_ITERS=100 immediately** — may explain why some experiments underperformed.

## Priority queue
### For 1xH100 (current screening):
1. ✅ Sliding window eval — DONE
2. **WARMDOWN_ITERS=100** (PR #94 fix, free improvement)
3. Doc-isolated eval (free -0.011)

### For 8xH100 (final submission):
4. **int6 STE QAT** — fake-quantize in CastedLinear.forward():
   ```python
   # In CastedLinear.forward():
   w = self.weight.to(x.dtype)
   scale = w.abs().amax(dim=1, keepdim=True) / 31.0
   w_q = (w / scale).round().clamp(-31, 31) * scale
   w = w + (w_q - w).detach()  # STE: gradients flow through, forward sees quantized
   ```
   Reduces quant penalty from ~0.05 to ~0.001 BPB.
5. **MLP 3x** expansion (MLP_HIDDEN=1536, with int6 to fit under 16MB)
6. **ROPE_BASE=200000** (PR #120, ~0.002 BPB gain vs default 10000)
7. MTP auxiliary head (training-only gradient enrichment)
8. SWA over checkpoints during warmdown
9. zstd-22 compression instead of zlib
10. Sliding window stride=64 (eval, ~0.012 over stride=512)

### Competition frontier (2026-03-19 22:30)
- **Without val-only**: ~1.157 BPB (PR #114)
- **With val-only**: 0.959 BPB (PR #120, by Devin AI)
- 120 PRs total, 0 merged. Winning meta: int6 STE + MLP 3x + seq4096 + sliding window + fp16 embed
