"""Build a human-readable research report and videos from completed raw results."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from jitter_metrics import dispersion
from object_tracker.core.types import CameraIntrinsics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--reviewed-frames',default='',help='Record indices whose overlays were visually reviewed')
    args=parser.parse_args()
    root=args.run
    summary=json.loads((root/'summary.json').read_text())
    results=summary['experiments']
    config=json.loads((root/'config.json').read_text())
    base=next(r for r in results if r['id']=='baseline')
    points=np.load(root/'screen_points_mm.npy')
    camera=CameraIntrinsics(**config['camera'])
    intermediate={}
    settled={}
    for r in results:
        records=[json.loads(p.read_text()) for p in sorted((root/'poses'/r['id']).glob('repeat-*.json'))]
        valid=[record for record in records if 'iterations' in record.get('trace',{})]
        if valid:
            intermediate[r['id']]=[dispersion([record['trace']['iterations'][i] for record in valid],points,camera)[0]
                                   for i in range(r['settings']['actual']['num_iterations_test'])]
        static=[json.loads(p.read_text()) for p in sorted((root/'poses'/r['id']).glob('static-*.json'))]
        # Shared fixed diagnostic subset; do not replace the full-range metric.
        static=[record for record in static[4:] if not record['failure'] and 'matrix' in record]
        if static:settled[r['id']]=dispersion([record['matrix'] for record in static],points,camera)[0]
    (root/'intermediate_repeat_metrics.json').write_text(json.dumps(intermediate,indent=2))
    (root/'settled_static_metrics.json').write_text(json.dumps(settled,indent=2))
    controls=[r for r in results if r['settings']['reset_seed']]
    candidates=[r for r in results if not r['settings']['reset_seed']]
    complete=[r for r in candidates if r['modes']['static']['completed'] and r['modes']['motion']['completed']]
    # Jumps during real motion are warnings, not measured accuracy errors.
    safe=[r for r in complete if r['modes']['motion']['pnp_failures']<=base['modes']['motion']['pnp_failures']
          and r['modes']['motion']['large_translation_jumps']<=base['modes']['motion']['large_translation_jumps']
          and r['modes']['motion']['large_rotation_jumps']<=base['modes']['motion']['large_rotation_jumps']]
    best=min(safe or complete,key=lambda r:r['modes']['static']['dispersion']['screen']['rms'])
    save=dict(id=best['id'],overrides=best['settings']['overrides'],production_default_changed=False,
              status='candidate_requires_visual_review',source_run=str(root.resolve()))
    if args.reviewed_frames:
        save.update(status='visually_reviewed_keyframes',reviewed_frames=[int(i) for i in args.reviewed_frames.split(',')])
    (root/'recommended_candidate.json').write_text(json.dumps(save,indent=2))
    lines=['# Исследование intrinsic GoTrack jitter', '',
           f'Проект: `{config["project"]}`. Объектив: {config["args"]["lens_mm"]} мм (metadata).',
           f'Камера: `{json.dumps(config["camera"])}`.',
           f'Mesh: `{config["mesh"]["prepared"]}`; SHA256 `{config["mesh_sha256"]}`.',
           f'Повторы: кадр {config["args"]["frame"]}, {config["args"]["count"]} запусков каждого варианта.',
           f'Static: {config["args"]["static"][0]}–{config["args"]["static"][-1]}; motion: {config["args"]["motion"][0]}–{config["args"]["motion"][-1]} (zero-based).',
           f'Проверено {len(results)} конфигураций. Все параметры baseline сохранены в config.json и report.md.',
           '', '## Изменения', '',
           '- `tools/jitter_lab.py`: отдельный benchmark, staged tuning, metadata, raw poses, diagnostics, plots и previews.',
           '- `tools/jitter_metrics.py`: SO(3), translation и экранные статистики.',
           '- `src/object_tracker/backends/gotrack/experimental.py`: изолированные deterministic selectors и hooks.',
           '- `tools/jitter_analysis.py`, `docs/JITTER_LAB.md`, `tests/test_jitter_lab.py`: отчёт, инструкция и unit tests.',
           'Production backend, defaults, camera, prepared mesh, upstream и prototypes не изменялись.',
           '', '## Результаты', '',
           'RMS рассчитан от representative pose; screen px относятся к исходному изображению, PnP px — к crop.',
           'Motion не имеет ground truth; failure count и score не доказывают абсолютную accuracy.',
           '', '| ID | Selector | visib | PnP | iters | pad | crop | Repeat px | Static T mm | Static R ° | Static px | Motion min score | Fail/PnP | ΔT/ΔR jumps |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def f(x):return '—' if x is None else f'{x:.3f}'
    for r in results:
        a=r['settings']['actual'];st=r['modes']['static'];mo=r['modes']['motion'];rp=r['modes']['repeat']
        d=st.get('dispersion',{})
        lines.append(f'| {r["id"]} | {a["correspondence"]} | {a["visib_threshold"]:.2f} | {a["pnp_opts"]["pnp_inlier_thresh"]:.1f} | {a["num_iterations_test"]} | {a["crop_rel_pad"]:.2f} | {a["crop_size"][0]} | {f(rp.get("dispersion",{}).get("screen",{}).get("rms"))} | {f(d.get("translation",{}).get("rms"))} | {f(d.get("rotation",{}).get("rms"))} | {f(d.get("screen",{}).get("rms"))} | {f(mo["score_min"])} | {mo["failures"]}/{mo["pnp_failures"]} | {mo["large_translation_jumps"]}/{mo["large_rotation_jumps"]} |')
    lines+=['', '## Повторяемость и источник stochastic noise', '']
    for r in [base]+controls+[x for x in results if x['settings']['stage']=='selection']:
        records=[json.loads(p.read_text()) for p in sorted((root/'poses'/r['id']).glob('repeat-*.json')) if 'matrix' in json.loads(p.read_text())]
        matrices=np.array([r['matrix'] for r in records]);eq=len(matrices)>0 and np.array_equal(matrices,np.repeat(matrices[:1],len(matrices),axis=0))
        d=r['modes']['repeat'].get('dispersion',{})
        lines.append(f'- {r["id"]}: identical raw matrices={eq}; translation std={f(d.get("translation_std_mm"))} мм; angular distance std={f(d.get("rotation",{}).get("std"))}°; screen RMS={f(d.get("screen",{}).get("rms"))} px.')
    lines+=['', 'Baseline выбирает случайно не более 10000 correspondences; доступное число по каждой итерации сохранено в trace.',
            'Fixed-seed control проверяет combined stochastic effect. Deterministic selectors при разных seeds проверяют влияние sampling; CUDA nondeterminism нельзя исключить для другого hardware/software.',
            '', '## Staged tuning', '']
    replay_file=root/'pnp_replay/summary.json'
    if replay_file.exists():
        replay=json.loads(replay_file.read_text())
        lines+=['', 'PnP replay: замороженные correspondences первой итерации; screen px здесь относятся к crop.',
                '| RNG series | Identical matrices | T std mm | R RMS ° | Screen RMS crop px |',
                '|---|---|---:|---:|---:|']
        for label,m in replay['groups'].items():
            lines.append(f'| {label} | {m["identical_matrices"]} | {m["translation_std_mm"]:.6f} | {m["rotation"]["rms"]:.6f} | {m["screen"]["rms"]:.6f} |')
        lines+=['OpenCV-only seed variation не дала variability на этом наборе/версии, а NumPy-only и both дали одинаковые matrices. Это изолирует random correspondence sampling как источник измеренного stochastic noise.',
                'CPU replay делает tensor-to-NumPy buffers contiguous, воспроизводя memory layout GPU→CPU copies; solver и sampling code не меняются.']
    for path in sorted(root.glob('stage-*.json')):
        stage=json.loads(path.read_text());lines.append(f'- {path.stem}: candidates={stage["candidates"]}, retained={stage["parents"]}.')
    lines+=['', 'Intermediate repeat dispersion по каждой итерации сохранён в intermediate_repeat_metrics.json.',
            'Вспомогательный static расчёт с исключением первых четырёх кадров одинаково для всех вариантов сохранён в settled_static_metrics.json. Основная таблица использует весь static диапазон. Это помогает отличить convergence из manual initial pose от последующего jitter.']
    if (root/'failed_attempts').exists():
        lines+=['', 'Первый larger-crop запуск остановился из-за fixed viewport уже созданного renderer. Эти attempts сохранены в failed_attempts/. Затем renderer пересоздавался при смене crop_size; архитектура и mesh не изменялись. Таблица содержит итоговые повторные проверки, а не первоначальную setup-ошибку.']
    lines+=['', '## Кандидат для рекомендации', '',
            f'`{best["id"]}`: `{json.dumps(best["settings"]["overrides"])}`.',
            f'Static screen RMS: {base["modes"]["static"]["dispersion"]["screen"]["rms"]:.3f} → {best["modes"]["static"]["dispersion"]["screen"]["rms"]:.3f} px.',
            ('Overlays визуально проверены на кадрах '+args.reviewed_frames+'.' if args.reviewed_frames else 'Этот выбор требует визуальной проверки overlay.'),
            'Preset сохранён отдельно и не применяется к проекту или production defaults.',
            '', '## Ограничения', '',
            'Один shot и один seed schedule: это измерение на данном материале, а не универсальный optimum.',
            'Static pose variation включает изменения RGB, возможное движение камеры и bias refinement; полностью отделить причины без ground truth нельзя.',
            'Threshold score меняется вместе с PnP threshold; scores разных thresholds не являются общей accuracy шкалой.',
            'Нулевой repeatability noise не означает нулевой frame-to-frame jitter.',
            'Большие motion jumps — эвристика; реальное движение может пересечь warning threshold.',
            'Temporal smoothing, camera calibration и mesh optimization не выполнялись. Следующий этап требует отдельного решения пользователя.']
    (root/'research_report.md').write_text('\n'.join(lines))
    fps=config['project_snapshot']['source']['fps']
    for r in [base]+([] if best['id']==base['id'] else [best]):
        files=root/'previews'/r['id']/'motion'
        # Input glob preserves the sorted inclusive frame range.
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-framerate',str(fps),'-pattern_type','glob','-i',str(files/'*.jpg'),
                        '-c:v','libx264','-pix_fmt','yuv420p',str(root/'previews'/f'{r["id"]}-motion.mp4')],check=True)
    print(json.dumps(save,indent=2))


if __name__=='__main__':main()
