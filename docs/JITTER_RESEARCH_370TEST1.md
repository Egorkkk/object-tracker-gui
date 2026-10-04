# GoTrack jitter: 370test1, объектив 40 мм

Исследование выполнено на активном проекте `/mnt/e/POL/ASSETS/EP08/0370/370test1`.
Использован его prepared phone mesh 80×160×11 мм. Camera: 1920×1080,
fx=fy=2133.3333, cx=960, cy=540, distortion=None. Camera, mesh и source hashes
проверены после тестов: изменений нет. Production defaults не менялись.

## Реально выполненные тесты

30 конфигураций: baseline, fixed-seed control и staged selection/confidence/PnP/
iterations/padding/crop. Каждая: 20 повторов кадра index 2, static 2–25 и
motion 26–70. Это 2670 raw results. Индексы zero-based: исходные PNG
0003–0026 для static, 0027–0071 для motion. Motion включает руку, захват,
подъём и частичный выход за край. Все итоговые конфигурации завершили диапазоны;
tracking failures и PnP failures — 0. Ground truth отсутствует.

Baseline: random, visib=0.30, PnP=2.0 crop px, RANSAC=3000,
confidence=0.999, max correspondences=10000, iterations=5, pad=0.10,
crop=280×280, re_crop_every_iter=True, gray background, ssaa=1.0.
Остальные actual options записаны в config.json и report.md.

## Основные результаты

Static dispersion — RMS относительно median translation / mean SO(3) rotation.
Screen RMS вычислен по фиксированным mesh vertices в pixels исходного RGB.

| Вариант | T RMS мм | R RMS ° | Screen RMS px | Median runtime с | Motion mean/min score | ΔT>50 мм / ΔR>30° |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.794 | 0.285 | 1.036 | 1.041 | 0.947 / 0.782 | 1 / 0 |
| top_confidence, 5 iterations | 0.569 | 0.220 | 0.856 | 0.971 | 0.921 / 0.707 | 0 / 0 |
| top_confidence, 2 iterations | 0.490 | 0.246 | 0.905 | 0.417 | 0.931 / 0.823 | 0 / 0 |
| random, 3 iterations | 0.579 | 0.250 | 0.984 | 0.622 | 0.942 / 0.789 | 0 / 0 |
| random, pad=0.15 | 0.672 | 0.217 | 0.923 | 0.957 | 0.963 / 0.863 | 0 / 0 |
| spatial_confidence | 0.779 | 0.213 | 0.887 | 1.153 | 0.932 / 0.758 | 0 / 2 |

Baseline static score mean/min: 0.991 / 0.982; top-confidence: 0.968 / 0.953.
Score — correspondence weighted inlier fraction, не absolute accuracy.
Большие jumps — warnings, реальное движение тоже может пересечь threshold.
Overlays baseline, top-confidence 5 и 2 iterations проверены на кадрах
2, 42, 50, 54, 62, 70; явного срыва трекинга не обнаружено.

## Ответы на вопросы исследования

1. **Stochastic при одинаковом input:** baseline repeat T std=0.844 мм,
   angular-distance std=0.0716°, RMS angle=0.237°, screen RMS=0.980 px.
2. **Источник:** first-iteration frozen PnP replay по 20 calls в четырёх
   RNG сериях. NumPy-only: T std=0.433 мм, R RMS=0.196°, screen RMS=0.197
   **crop px**. OpenCV-only: все matrices совпали. Both дал те же matrices,
   что NumPy-only. В этом наборе/версии variability создаёт sampling.
3. **Deterministic selection:** top и spatial дают identical matrices во
   всех 20 full-refinement repeats, хотя seeds меняются. Fixed-seed random
   control также даёт identical matrices. Малые остатки screen RMS порядка
   1e-5 px — численная разница float32 rotation и SO(3) reference.
4. **Confidence threshold:** 0.30/0.40/0.50/0.60 с top selector дали одинаковый
   static RMS 0.856 px. Оставить 0.30; повышение ради static jitter не оправдано.
5. **PnP:** 2.0 crop px лучше 1.0 и 1.5 для обоих исследованных selectors.
   Random: 1.532 / 1.231 / 1.036 px; top: 1.007 / 1.003 / 0.856 px.
6. **Iterations:** для top лучше 5 по screen jitter; 2 — быстрый компромисс.
   Утверждение, что лишние iterations обязательно добавляют noise, не подтвердилось.
   Random 3 немного лучше random 5; intermediate metrics сохранены отдельно.
7. **Padding:** random 0.15 улучшает jitter; top лучше оставить 0.10.
   0.05 ухудшает оба selectors; random даёт 2 warnings, top — 1.
8. **Larger crops:** 336 и 420 выполняются без изменения архитектуры, но
   не помогают при остальных фиксированных settings. Random: 1.931 / 2.613 px;
   top: 1.022 / 1.828 px. При смене размера нужен новый renderer, поскольку
   viewport GoTrack фиксирован. Первые ошибки и повторные проверки сохранены.
   PnP=2 crop px при larger crop меняет effective original-image tolerance;
   это результат данной конфигурации, а не доказательство бесполезности
   larger crops при любых PnP settings.
9. **Лучший static кандидат:** top-confidence 5, остальные baseline settings:
   screen jitter ниже на 17.3%, translation RMS — на 28.3%, rotation RMS — на 22.7%.
10. **Robustness:** motion 26–70 завершён без failures и warning jumps у этого
    кандидата; score ниже baseline. Визуальная проверка ключевых overlays не
    выявила потери объекта. Без ground truth absolute accuracy не установлена.

## Рекомендация

Experimental preset: `correspondence=top_confidence`, `visib_threshold=0.30`,
`pnp_inlier_thresh=2.0`, `num_iterations_test=5`, `crop_rel_pad=0.10`,
`crop_size=[280,280]`; прочие параметры baseline. Это e002.
Быстрый вариант e018: только iterations=2 — screen RMS 0.905 px при 0.417 с/кадр.
Ни один preset не применяется автоматически.

Результат относится к одному shot и одному seed schedule. Static variation
содержит влияние RGB и возможное движение камеры; нулевой repeated-input
noise не означает отсутствия frame-to-frame jitter. Дополнительный static
расчёт без первых четырёх convergence frames не подменяет основную таблицу.
Temporal filtering, camera calibration и mesh optimization не выполнялись.

## Артефакты и реализация

Run: `outputs/jitter_lab/370test1-40mm-v2/`.
Полная таблица и lineage: `research_report.md`, `report.md`, `results.csv`,
`summary.json`, `stage-*.json`. Poses/intermediate/PnP diagnostics: `poses/`.
Plots: `plots/`. Preview videos: `previews/baseline-motion.mp4`,
`previews/e002-motion.mp4`. Replay: `pnp_replay/`. Integrity: `integrity_check.json`.
Checkpoint и DINO SHA256, versions, commits и source snapshots сохранены.

Добавлены tools/jitter_lab.py (runner), tools/jitter_metrics.py (метрики),
tools/jitter_pnp_replay.py (RNG isolation), tools/jitter_analysis.py (отчёт),
backends/gotrack/experimental.py (селекторы и scoped hooks), unit tests и
инструкция docs/JITTER_LAB.md. Production backend и upstream не редактировались.
35 unit tests прошли, включая 8 tests selectors/метрик/восстановления hooks.
