# Parameter G0lf — Autonomous Research Agent Results

Live dashboard: **https://nglain.github.io/g0lf-results/**

## Current Best

| Metric | Value |
|--------|-------|
| **Best BPB** | **1.2075** (exp_033) |
| Official baseline | 1.2244 |
| **Baseline beaten** | Yes (-0.017) |
| Total experiments | 35 |
| Keeps / Discards | 18 / 17 |

## Approach

Two-phase autonomous ML research system using AI coding agents on RunPod GPUs.

**Phase 1: Calibrate search method.** Tested statistical approaches (Taguchi L8 orthogonal arrays, Bayesian optimization, successive halving) against known results from competition PRs. Taguchi L8 with fixed-seed paired comparison (delta detection ±0.001 BPB) proved most cost-effective.

**Phase 2: Find optimal combination.** Two agents work in parallel:
- **ADVISOR**: monitors competition, analyzes experiment history, synthesizes hypotheses, writes strategy to shared `advice.md`
- **EXECUTOR**: implements techniques, runs GPU experiments using calibrated statistical method, documents results

Communication through file-based bus: `advice.md` (strategy), `experiments.json` (results), `knowledge.md` (accumulated insights).

## Experiment History

### Phase 1: A40 Screening (exp_001–003)
- Established that step speed dominates on limited compute
- SwiGLU adds params → slower → worse at short training

### Phase 2: 1×H100 Architecture Search (exp_004–028)
- Layer sweep: optimal = 3L for 2-min, 6-9L for 10-min
- LR sweep: matrix_lr 0.08-0.12 optimal for 2-min
- Muon tuning needs 7000+ steps — doesn't work on 1×H100
- int6 STE conflicts with Muon optimizer
- seq4096 marginal on 1×H100 (not enough steps to benefit)
- Best 1×H100: 1.348 BPB

### Phase 3: 8×H100 Full Compute (exp_029–035)
- Competition Muon tuning works with 7000 steps: -0.005 BPB
- ROPE_BASE=200K: additional -0.003 BPB
- seq4096 confirmed beneficial with enough steps
- int6 STE still conflicts with Muon
- Larger batch (786K) = fewer steps = worse
- **Best: 1.2075 BPB** (exp_033)

### Phase 4: Taguchi L8 Combinatorial Search (in progress)
- Testing 7 techniques simultaneously via orthogonal array
- MLP3x, OrthoInit, BigramHash, SmearGate, fp16 embed, Primer-EZ, int6 post-quant

## Key Insights

1. **Compute regime changes everything**: A40/2min, 1×H100/10min, 8×H100/10min have different optima
2. **Techniques can conflict**: int6 STE + Muon = +0.007 worse (not additive!)
3. **Statistical DoE > blind search**: Taguchi L8 tests 7 factors in 8 runs instead of 128
4. **Paired seed comparison**: SEED=1337 reduces noise from ±0.005 to ±0.001
5. **Competition monitoring is essential**: research scout found proven techniques faster than blind exploration

## Files

| File | Description |
|------|-------------|
| `experiments.json` | All experiment results (structured log) |
| `knowledge.md` | Accumulated insights + competition intel |
| `train_gpt.py` | Current best model code |
| `runs/` | Full training logs for each experiment |
| `index.html` | Live dashboard (GitHub Pages) |

## Rules

- **No val-only training** — honest training on real data only
- All experiments reproducible with fixed seeds
- Results auto-synced to this repo by the agent system

## Cost

| Phase | GPU | Cost | Experiments |
|-------|-----|------|-------------|
| A40 screening | 1×A40 | ~$3 | 14 |
| 1×H100 search | 1×H100 PCIe | ~$12 | 15 |
| 8×H100 validation | 8×H100 SXM | ~$25 | 6+ |
| **Total** | | **~$40** | **35+** |

## Links

- Challenge: [OpenAI Parameter Golf](https://openai.com/index/parameter-golf/)
- Dashboard: [nglain.github.io/g0lf-results](https://nglain.github.io/g0lf-results/)
