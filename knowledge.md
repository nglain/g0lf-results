# Knowledge Base

## Текущая гипотеза
exp_015: Увеличить model_dim до 768 при 3 layers + batch=131K. Больше width может помочь при shallow model.

## Лучший результат
- **exp_013 (3 layers, batch=131K): 1.5700 bpb** (A40, 2 мин, 724 шага, ~2.8MB compressed)

## Текущие defaults в train_gpt.py
- 3 layers, model_dim=512, num_heads=8, num_kv_heads=4
- batch=131,072 tokens
- Всё остальное — оригинальный baseline

## Что работает
- **Меньше layers** (fewer layers = faster steps = more training): 9→6→4→3→2 все лучше baseline
- **Меньше batch** (131K оптимум): 524K→262K→131K все улучшают. 131K — sweet spot.
- **3 layers лучше 2** при batch=131K (1.5700 vs 1.5839) — есть оптимум capacity vs speed

## Что не работает
- **SwiGLU**: +params → slower → worse на ограниченном compute
- **Увеличение dim (768 vs 512)**: при 2 layers, dim=768 хуже (2.39 vs 1.94) — too slow
- **Batch < 131K**: 65K и 98K хуже — слишком шумные градиенты
- **4 layers + batch=131K**: хуже чем 3 layers (1.59 vs 1.57)

## Паттерны и инсайты
- На A40/2мин: **speed is king**. Все улучшения пока от увеличения числа шагов.
- Оптимум layers: 3 при batch=131K (проверено 2,3,4)
- Оптимум batch: 131K при 3 layers
- Compressed size ~2-3MB — огромный запас до 16MB
- ВАЖНО: На H100/10мин оптимум будет другой (больше layers, больше batch)

## Комбинации на проверку
- 3 layers + dim=384 (ещё быстрее, больше шагов?)
- 3 layers + dim=256 (ещё меньше?)
- 3 layers + уменьшить warmup (20→5?)
- 3 layers + увеличить LR (быстрее сходимость)
- Looped layers: 3 unique → 6 effective
