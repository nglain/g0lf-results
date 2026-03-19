# Knowledge Base

## Текущая гипотеза
exp_021: Try seq_len=2048 on 9L — balance between context and total tokens on 1xH100.

## Лучший результат
- **exp_018/020 (9L, seq4096, H100/10min): 1.3525 bpb** (887 steps, 676ms/step, 11.9MB)
- **exp_017 (6L, seq1024, H100/10min): 1.3573 bpb** (1492 steps, 402ms/step, 9.5MB)
- Target baseline: **1.2244 bpb** (8xH100, 10min) — gap: 0.128

## Current train_gpt.py state
- Defaults: 3 layers, dim=512, lr=0.10 (for quick screening)
- Best 10min config: 9L/seq4096/lr=0.04 → 1.3525
- Has: looped layers (off), sliding window eval (via SLIDE_STRIDE env), forward_logits method
- NO val-only training (forbidden by human)

## Что работает (H100/10min)
- **9L, 10min**: 1.358 bpb (seq1024). Best layer count for 1xH100.
- **6L, 10min**: 1.357 bpb (seq1024). Nearly identical but only 9.5MB (vs 12.4MB). More size room.
- **seq4096**: 1.353 bpb. Marginal -0.006 on 1xH100. Should help more on 8xH100.
- Sliding window eval: implemented, awaiting 8xH100 for proper testing.

## Что не работает
- **Competition Muon tuning on 1xH100**: lr=0.02/mom=0.99/warmdown=3000 → 1.414 bpb (WORSE). Needs 9500 steps, we have ~890.
- **12L at 10min**: 1.373 (worse than 9L). Too slow.
- **SwiGLU (A40)**: failed, but top PRs don't use it either. Skip.
- **Wider dim**: failed on 2-min AND competition uses 512. Skip.
- **Looped layers**: failed, PR #31 confirmed worse. Skip.
- **Smaller batch (262144)**: worse.

## Compression awareness
- 6L/seq1024: 9.5MB (6.5MB headroom)
- 9L/seq1024: 12.4MB (3.6MB headroom)
- 9L/seq4096: 11.9MB (4.1MB headroom)
- Budget: 16MB total

## 1xH100 vs 8xH100 reality
- 1xH100: ~890 steps at 9L/seq4096. Competition uses ~9500 steps on 8xH100. 10x gap.
- Competition Muon tuning, sliding window eval, and larger architectures are all tuned for 8xH100.
- Our role: find the best ARCHITECTURE and CODE. Training quality will improve with 8xH100 final.
- Focus on techniques that help at ANY step count, not just high-step regimes.

## Competition intel (honest techniques only)
- seq4096: universal (✓ implemented)
- Sliding window eval: -0.03-0.05 bpb FREE (✓ implemented, need 8xH100 to test)
- MLP 3x expansion: PR #70 uses (try this)
- fp16 tok_emb: saves space (try this)
- int6 middle layers: saves ~1.6MB → fits 10th layer (try this)
- NTK RoPE eval: PR #60 (try this)
- warmdown quant scheduling: PR #61 (try this)

## Priority queue
1. seq_len=2048 (balance context vs total tokens on 1xH100)
2. MLP 3x expansion (PR #70 approach)
3. fp16 tok_emb (smaller compressed size → room for more layers)
4. NTK RoPE at eval (free improvement like slide eval)
5. int6 middle layers + 10th layer
