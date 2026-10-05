# GoTrack jitter lab

Исследование запускается отдельно от UI. `GoTrackBackend` остаётся единственным
inference pipeline; модель загружается один раз. Исходный проект, камера,
prepared mesh и upstream GoTrack открываются только для чтения.

```bash
MPLCONFIGDIR=/tmp/jitter-matplotlib bash scripts/run_python.sh tools/jitter_lab.py \
  --project /path/to/project.json \
  --output outputs/jitter_lab/run_name \
  --frame 2 --static 2:25 --motion 26:70 --count 20 --lens-mm 40
```

`--lens-mm` — только metadata. fx/fy/cx/cy берутся из проекта без пересчёта.
На первом кадре должна существовать draft/anchor/pose. Static начинается с
этого кадра, motion продолжает static без разрыва. Диапазоны включительные,
индексы zero-based; номера исходных файлов сохранены в snapshot проекта.
Проект не сохраняется, журнал poses не восстанавливается и не изменяется.

Можно выбрать этапы: `--stages selection,visib,pnp,iterations,pad,crop`.
Пустое `--stages ''` выполняет baseline и baseline с фиксированным seed.
`--survivors 1` (или 2) оставляет baseline и лучших кандидатов предыдущего
этапа. Сначала сравнивается completion, failures, большие скачки, PnP failures,
затем static screen RMS. Это автоматический shortlist, не доказательство
accuracy. Каждый кандидат проверяется на repeat, static и motion; полный
Cartesian product не выполняется. Lineage и выбор после каждого этапа
сохраняются в `stage-*.json`.

`--resume` повторно использует завершённые experiment summaries. Незавершённый
кандидат запускается заново. Проверяются baseline, camera, mesh hash, initial
pose, frame hashes и параметры run; metadata продолжения сохраняются отдельно.
Кандидаты с одинаковыми **фактическими** настройками не запускаются повторно,
даже если explicit overrides отличаются.
`--resume --retry-failed` архивирует прошлые error attempts и повторяет их.

Random сохраняет upstream selection и solver. Confidence selection реализован
guarded substitution только блока sampling в in-memory копии функции PnP.
Остальной upstream код solve/evaluation/fallback тот же. При изменении
ожидаемого sampling-блока hook останавливается с ошибкой. Spatial selection
использует 8×8 grid по bounding box target correspondences: stable confidence
ordering внутри ячеек, затем round-robin по рангу. Это повышает coverage,
но само по себе не гарантирует меньший jitter.

Research hooks допустимы **только в отдельном однопоточном процессе**. В `finally`
восстанавливаются model options и обе обёрнутые функции. UI использует только
sampling context в единственном tracking worker: checkbox в Active Tracking
Solution сохраняет `backend.settings.correspondence_selector`; без поля применяется
`random`, при включении — `top_confidence`. Остальные model options не меняются.
CUDA deterministic algorithms не
включаются; seeds Python/NumPy/Torch/OpenCV задаются перед каждым кадром.
Baseline использует разные, воспроизводимые seeds для повторов; fixed-seed
control сбрасывает все generators перед каждым запуском.

`config.json` фиксирует actual baseline (включая NamedTuple defaults и
effective PnP options), checkpoint SHA256, commits, source runtime hash,
снимок проекта, frame hashes, camera, prepared mesh hash, initial pose,
seed и hashes lab source files. `poses/` содержит исходные matrices, scores,
runtime, PnP diagnostics и intermediate poses каждой итерации.
`results.csv`, `summary.json`, `report.md` и `plots/` создаются после каждого
кандидата. Preview JPEG contours сохраняются для каждого кадра sequence.

Метрики static/repeat: median translation, mean rotation на SO(3), расстояния
от reference pose, RMS/median/MAD/p95/std и translation std по осям. Экранная
метрика использует до 1000 фиксированных mesh vertices (сохранены в NPY),
общие точки с positive depth, без screen clipping. Rotation std — разброс
величины угла от reference; RMS также обязателен. Ни representative pose,
ни статистика не подменяют сохранённые raw poses и не сглаживают их.

Motion: completion, errors, score mean/min, warnings, скачки более 50 мм /
30° (thresholds configurable). Нет ground truth: движение нельзя трактовать
как jitter, а score не является абсолютной точностью. Требуется визуальная
проверка overlays. PnP failures отдельно учитывают insufficient points и
нулевой quality: upstream `failed` недостаточен для диагностики этих случаев.

PnP reprojection threshold измеряется в **crop pixels**; screen-space jitter —
в pixels исходного изображения. При смене crop_size 2 crop px означают другой
угловой допуск. Score — weighted inlier fraction и зависит от threshold,
поэтому scores с разными PnP thresholds не являются общей accuracy-шкалой.

Crop 336 и 420 — experimental. Оба кратны patch size 14 и имеют чётную feature
grid; их работоспособность проверяется реальным forward, без изменения
архитектуры. Ошибка формы/OOM сохраняется и считается failure; pipeline не
подгоняется под новый crop.
При смене crop_size пересоздаётся renderer: upstream renderer фиксирует
viewport первого render и запрещает менять его размер. Mesh и камера при
этом остаются прежними; модель/архитектура не меняются.

Unit tests:

```bash
bash scripts/run_python.sh -m unittest discover -s tests -p 'test_jitter_lab.py' -v
```

Дополнительная изоляция NumPy sampling и OpenCV RNG на замороженных
correspondences **первой** refinement iteration:

```bash
bash scripts/run_python.sh tools/jitter_pnp_replay.py outputs/jitter_lab/run_name
bash scripts/run_python.sh tools/jitter_analysis.py outputs/jitter_lab/run_name
```

Replay загружает GoTrack один раз, сохраняет tensor inputs PnP в NPZ и
выполняет четыре серии по 20 CPU PnP calls: fixed seeds, только NumPy меняется,
только OpenCV меняется, оба меняются. Здесь screen RMS выражен в **crop px**,
поскольку сохранён crop camera и poses этой итерации. Результат нельзя
напрямую сравнивать с output full refinement в pixels исходного RGB.
`--from-frozen` повторяет CPU replay из сохранённого NPZ без загрузки GPU модели.
CPU tensor-to-NumPy buffers делаются contiguous: strided CPU tvec OpenCV
не принимает как output buffer, тогда как GPU→CPU copy в обычном pipeline
уже даёт contiguous layout. Sampling и solver алгоритмы не меняются.

Default preset остаётся прежним независимо от результатов. Рекомендация
experimental preset оформляется отдельно после просмотра результатов.
