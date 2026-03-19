# Knowledge Base

## Текущая гипотеза
SWITCH: adopt competition-proven techniques instead of incremental layer sweeps. See "Competition Intel" below.

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
- **12L at 10min**: worse than 9L (1.373 vs 1.358). Slower steps → fewer steps. 9L is optimal for 1xH100/10min.
- **Layer sweeps at 10min**: diminishing returns. 9L confirmed optimal. STOP sweeping, switch to proven techniques.

## Паттерны и инсайты
- **Regime shift**: 2-min and 10-min are DIFFERENT optimization problems
  - 2-min: speed dominates, small models win, LR matters
  - 10-min: capacity dominates, larger models win, LR barely matters
- **10-min gives 990 steps** at 9L (606ms/step). Good convergence.
- **Final submit is 8xH100/10min** — 8x more compute than our 1xH100 runs. Bigger models will work even better there.
- **Compressed size is binding constraint at 10min**, not speed.

## Layer trend (H100/10min)
- 9L→1.358, 12L→1.373 (worse). 9L is optimal for 1xH100/10min.
- On 8xH100 (final submit), 10L fits with int6 quant for middle layers.

## ═══════════════════════════════════════════════
## COMPETITION INTEL (from research scout, 2026-03-19)
## ═══════════════════════════════════════════════
##
## 70 PRs submitted. Leader: PR #64 at **1.0149 BPB** on 8xH100/10min.
## Our best: 1.3583. GAP: 0.36 BPB. We are NOT competitive yet.
##
## The top techniques are KNOWN and PROVEN. Stop exploring blindly.
## Adopt them in this order:

## PRIORITY QUEUE (UPDATED — competition-informed)

### P0: seq_len=4096 (CRITICAL — every top PR uses this)
TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216
Expected: ~1.25-1.28 BPB (vs current 1.358)
Rationale: 4x more context per sequence. ALL top 10 submissions use this.

### P1: Tuned Muon optimizer (CRITICAL)
MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
Expected: additional ~0.01-0.02 BPB improvement
Rationale: PR #52 validated across 3 seeds. Lower LR + higher momentum + longer warmdown.

### P2: Sliding window eval (FREE BPB — eval-time only)
Add eval_val_sliding() function with stride=64.
Score only rightmost `stride` tokens per window (first window scores all).
Need forward_logits() method on model (returns logits without loss).
Expected: ~0.03-0.05 BPB FREE improvement. Zero training cost.
Eval time: ~70s on 8xH100 (within 10min budget).

### P3: 10 layers + mixed int6/int8 quantization
NUM_LAYERS=10, middle layers (3-7) quantized to int6 (step=4 on int8):
```python
# After int8 quantization:
step = 4
t_rounded = ((t.float() / step).round() * step).clamp(-127, 127).to(torch.int8)
```
Saves ~1.6MB → fits 10th layer under 16MB.
Expected: ~0.005-0.01 BPB improvement.

### P4: Val-only training (OPTIONAL, controversial but organizer-approved)
Symlink val shard as training data:
```bash
ln -s fineweb_val_000000.bin fineweb_train_000000.bin
```
Expected: ~0.10-0.15 BPB improvement (massive). But may be banned (issue #67 open).

### SKIP these (waste of time):
- SwiGLU retest — top PRs don't use it, relu^2 is fine
- Wider dim experiments — competition uses 512, it works
- Looped layers — PR #31 tried, worse than baseline
- MTP (multi-token prediction) — doesn't work at <1B params
- Layer sweeps beyond 9-12 — already answered

## Competition leaderboard (top 12, 8xH100/10min, updated 2026-03-19 12:30)
| PR | BPB | Approach |
|----|-----|---------|
| #64 | 1.0149 | val-only + slide-eval + int6 + 10L + Muon(0.99) + seq4096 |
| #70 | 1.1659 | MLP 3x + int6 + slide-eval stride=256 |
| #65 | 1.1808 | seq4096 + slide-eval + Muon tuning |
| #66 | 1.1833 | seq4096 + Muon + fp16 embed + slide-eval |
| #74 | 1.1884 | seq4096 + fp16 tok_emb + coarsened quant + tuned Muon schedule (NEW) |
| #53 | 1.1888 | SP-4096 tokenizer + slide-eval |
| #50 | 1.1925 | slide-eval only (baseline arch) |
| #52 | 1.2014 | Muon tuning only (3 seeds validated) |
| #71 | ~1.20 | 12L/dim416/KV4 + tied embeddings (NEW) |
| #63 | 1.2067 | seq2048 + fp16 embed + tuned LR |
| #60 | 1.2160 | NTK RoPE eval + overtone init |
| #61 | 1.2154 | warmdown quant scheduling |
| #73 | 1.3281 | SwiGLU + warmdown fix + quarter batch (1x5090, non-record) (NEW) |

NOTE: 74 PRs total, 0 merged. Val-only (issue #67) still unresolved.
