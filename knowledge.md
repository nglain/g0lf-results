# Knowledge Base

## Текущая гипотеза
exp_005: Try 4 layers on H100. On A40, fewer layers helped a lot. On H100 with ~406ms/step we get 296 steps — need to check if same trend holds or if capacity becomes bottleneck.

## Лучший результат
- **exp_004 (6 layers, H100): 1.7034 bpb** (296 steps, 406ms/step, 5.8MB compressed)
- **exp_003 (6 layers, A40): 2.7606 bpb** (109 steps, 1103ms/step, 4.2MB compressed)

## Baseline архитектура (train_gpt.py, current)
- **6 transformer layers**, model_dim=512, num_heads=8, num_kv_heads=4 (GQA)
- vocab_size=1024, seq_len=1024, tied_embeddings=True
- MLP: relu^2, 2x expansion
- ~406ms/step on H100, ~1103ms/step on A40

## Что работает
- **Меньше layers (6 vs 9)**: -0.436 bpb on A40 (73→109 steps)
- GQA fix (repeat_interleave) для PyTorch 2.4 compat

## Что не работает
- **SwiGLU (exp_002, A40)**: +0.067 bpb. More params = slower steps = fewer iterations on A40. May be worth retesting on H100.

## Тупики
(пока пусто)

## Паттерны и инсайты
- **H100 vs A40**: H100 gives ~4x more steps in same wall time (296 vs 73 for 6-layer config). Absolute bpb much better (1.70 vs 2.76).
- **На A40/2мин**: скорость шагов >> размер модели
- **На H100**: с 406ms/step и 296 steps, capacity may become bottleneck — larger models may help
- Compressed size: 5.8MB для 6 layers — огромный запас до 16MB

## Тренды
- A40 layers: 9→3.20, 6→2.76 (improving with fewer layers)
- H100 layers: 6→1.70 (need more data points: 4, 9)

## Комбинации на проверку
- 4 layers on H100 (faster steps?)
- 9 layers on H100 (more capacity, still enough steps?)
- 6 layers + larger dim (768) on H100
- 6 layers + SwiGLU on H100 (retry with more steps to amortize)
- Looped layers (4 unique × 2 repeats = 8 effective)
