# Offline pose sensitivity lab

Инструмент читает project.json напрямую, не создаёт Application/GoTrackBackend,
не запускает inference, не фильтрует и не записывает исходные poses.
Использует сохранённые static/motion записи из `poses/e002` предыдущего research.
Перед расчётом проверяет preset, camera/source metadata, SHA256 prepared mesh
и выбранных RGB; после — byte hashes project, всех track records, mesh и inputs.
Output запрещён внутри project/source/input research; непустой output отвергается.

```bash
MPLCONFIGDIR=/tmp/pose-sensitivity-mpl bash scripts/run_python.sh \
  tools/pose_sensitivity_lab.py \
  --project /mnt/e/POL/ASSETS/EP08/0370/370test1 \
  --track outputs/jitter_lab/370test1-40mm-v2/poses/e002 \
  --output outputs/pose_sensitivity/370test1 \
  --range 2:25 --frames 2,42,50,54,62,70 \
  --translation-mm 1 --rotation-deg 0.1 --near-fraction 0.25
```

Для нового запуска укажите новый пустой output. `--frames 2` выбирает одну pose;
`--range` задаёт inclusive ascending indices для adjacent deltas. `--max-vertices`
ограничивает фиксированный deterministic subset (по умолчанию 2000; здесь все
364 уникальные vertices, те же точки, что в предыдущей работе).
Не смешиваются независимые static/motion tracks: основной диапазон 2–25 целиком
относится к static; representative motion poses анализируются независимо.

Math: `T_cam_from_object`, OpenCV, mm. Camera rotation — left composition;
local rotation — right composition. Pivot — object origin при фиксированном t.
Relative angular changes — SO(3) rotation vectors, не Euler subtraction.
Split screen tests оставляют R_n/t_next или R_next/t_n; full использует pose_next.
Per-axis rotvec tests — first-order contributions; коррелированные вклады могут
компенсироваться. Никакого causal/absolute accuracy вывода без ground truth.

Результаты: `config.json`, `summary.json`, `report.md`, `integrity_check.json`,
`vertices_mm.npy`, четыре CSV в `tables/`, графики в `plots/`, nine DOF RGB
heatmaps каждого selected frame в `overlays/`. Цвет — истинный pixel displacement,
стрелки ×20. ± результаты и per-vertex vectors сохранены в CSV; overlays показывают +.
PCA covariance modes — дополнительная диагностика angular variation, не фильтр.
