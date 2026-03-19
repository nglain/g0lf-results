# Knowledge Base

## Текущая гипотеза
exp_004: Попробовать 4 layers. Если тренд сохраняется (меньше layers → больше шагов → лучше), можем найти оптимум.

## Лучший результат
- **exp_003 (6 layers): 2.7606 bpb** (A40, 2 мин, 109 шагов, 4.2MB compressed)

## Baseline архитектура (train_gpt.py, updated)
- **6 transformer layers** (было 9), model_dim=512, num_heads=8, num_kv_heads=4 (GQA)
- vocab_size=1024, seq_len=1024, tied_embeddings=True
- MLP: relu^2, 2x expansion
- ~1103ms/step on A40

## Что работает
- **Меньше layers (6 vs 9)**: -0.436 bpb. Больше шагов за то же время = лучше convergence на A40/2мин
- GQA fix (repeat_interleave) для PyTorch 2.4 compat

## Что не работает
- **SwiGLU (exp_002)**: +0.067 bpb. Увеличение params при ограниченном training time вредит.

## Тупики
(пока пусто)

## Паттерны и инсайты
- **КЛЮЧЕВОЙ**: На A40/2мин, скорость шагов >> размер модели. Меньше параметров = больше шагов = лучше convergence.
- Compressed size: 4.2MB для 6 layers — огромный запас до 16MB. Можно увеличивать dim позже.
- Тренд: 9 layers (73 steps, 3.20 bpb) → 6 layers (109 steps, 2.76 bpb). Стоит проверить 4 layers.
- ВАЖНО: оптимизации для A40/2мин (меньше модель) могут НЕ быть оптимальны для H100/10мин (больше compute). Нужно будет пересмотреть на финальном железе.

## Комбинации на проверку
- 4 layers (ещё быстрее?)
- 6 layers + увеличить dim (512→768) при том же param budget
- 6 layers + looped layers (6 unique → 12 effective)
