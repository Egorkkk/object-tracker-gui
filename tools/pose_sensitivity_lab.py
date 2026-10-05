"""Offline pose sensitivity lab: reuse verified e002 poses, never run inference.

Example:
  bash scripts/run_python.sh tools/pose_sensitivity_lab.py \
    --project /mnt/e/POL/ASSETS/EP08/0370/370test1 \
    --track outputs/jitter_lab/370test1-40mm-v2/poses/e002 \
    --output outputs/pose_sensitivity/370test1
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from object_tracker.core.mesh import load_mesh
from pose_sensitivity_metrics import (depth_split, displacement, perturb, project,
                                      relative_delta, rotation_modes, series_stats, split_pose)

AXES = 'XYZ'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_range(value):
    start, end = map(int, value.split(':'))
    if start >= end:
        raise argparse.ArgumentTypeError('Range must have >=2 ascending frames')
    return list(range(start, end+1))


def load_track(directory):
    records = {}
    files = {}
    for path in sorted(directory.glob('*.json')):
        record = read(path)
        if record.get('mode') not in ('static', 'motion'):
            continue
        index = record['frame']
        if record.get('experiment') != directory.name:
            raise ValueError(f'Pose belongs to a different candidate: {path}')
        if record.get('failure') or 'matrix' not in record:
            raise ValueError(f'Failed/missing pose: {path}')
        if index in records:
            raise ValueError(f'Ambiguous duplicate pose for frame {index}')
        matrix = np.asarray(record['matrix'], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])
                or not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=2e-5)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=2e-5)):
            raise ValueError(f'Invalid rigid pose: {path}')
        records[index], files[index] = matrix, path
    return records, files


def validate_inputs(args, state, config, candidate):
    actual = candidate['settings']['actual']
    expected = dict(correspondence='top_confidence', visib_threshold=.30,
                    num_iterations_test=5, crop_rel_pad=.10, crop_size=[280, 280])
    if any(actual[k] != v for k, v in expected.items()) or actual['pnp_opts']['pnp_inlier_thresh'] != 2.:
        raise ValueError('Track does not match requested deterministic e002 configuration')
    if str(args.project.resolve()) != str(Path(config['project']).resolve()):
        raise ValueError('Project differs from saved track provenance')
    if state['camera'] != config['camera'] or state['camera'].get('distortion') is not None:
        raise ValueError('Camera changed or distortion unsupported by pinhole diagnostic')
    if state['source'] != config['project_snapshot']['source']:
        raise ValueError('Source metadata differs from original research')
    if state['mesh'] != config['mesh'] or sha(state['mesh']['prepared']) != config['mesh_sha256']:
        raise ValueError('Prepared mesh differs from original research')
    if not np.allclose(state['mesh']['dimensions_mm'], [80, 160, 11], atol=1e-4):
        raise ValueError('Unexpected mesh dimensions')
    if args.translation_mm <= 0 or args.rotation_deg <= 0 or args.max_vertices < 2:
        raise ValueError('Perturbation magnitudes must be positive; >=2 vertices required')
    # Never permit output within the production project/source or input lab run.
    for protected in (args.project, Path(state['source']['path']), Path(state['mesh']['prepared']).parent,
                      args.track.parent.parent):
        if args.output.resolve().is_relative_to(protected.resolve()):
            raise ValueError(f'Output overlaps read-only input: {protected}')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be a new/empty directory; previous analyses are preserved')


def make_overlay(path, image, uv, shifted, depth, near, far, title, extent):
    delta = shifted-uv; distance = np.linalg.norm(delta, axis=1)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    scatter = ax.scatter(*uv.T, c=distance, s=16, cmap='inferno', vmin=0, vmax=extent)
    ax.scatter(*uv[near].T, facecolors='none', edgecolors='cyan', s=32, linewidths=.4, label='Nearest depth quartile')
    ax.scatter(*uv[far].T, facecolors='none', edgecolors='lime', s=32, linewidths=.4, label='Farthest depth quartile')
    chosen = np.arange(0, len(uv), max(1, len(uv)//45))
    ax.quiver(uv[chosen, 0], uv[chosen, 1], delta[chosen, 0]*20, delta[chosen, 1]*20,
              angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=.002)
    low, high = uv.min(axis=0)-30, uv.max(axis=0)+30
    ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1]); ax.set_aspect('equal')
    ax.set_title(title+'\nArrows amplified 20x; all fixed vertices, including hidden surface')
    ax.set_xlabel('Original RGB x (px)'); ax.set_ylabel('Original RGB y (px)')
    ax.legend(fontsize=7); fig.colorbar(scatter, ax=ax, label='Displacement (px)')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def sensitivity(args, state, poses, points):
    rows, vertex_rows = [], []
    definitions = [('camera', 'translation', i, args.translation_mm) for i in range(3)]
    definitions += [(space, 'rotation', i, args.rotation_deg) for space in ('camera', 'local') for i in range(3)]
    for frame in args.frames:
        matrix = poses[frame]; uv = project(points, matrix, state['camera'])
        near, far, depth = depth_split(points, matrix, args.near_fraction)
        overlays = []
        for space, kind, axis, step in definitions:
            dof = ('T' if kind == 'translation' else 'R')+AXES[axis]
            positive = None
            for sign in (-1, 1):
                shifted = project(points, perturb(matrix, kind, axis, sign*step, space), state['camera'])
                metric = displacement(uv, shifted, near, far)
                row = dict(frame=frame, space=space, dof=dof, sign=sign, step=step,
                           unit='mm' if kind == 'translation' else 'degree', **metric)
                row.update(rms_px_per_unit=metric['rms_px']/step,
                           rms_px_per_0_1_degree=metric['rms_px']/step*.1 if kind == 'rotation' else None)
                rows.append(row)
                if sign == 1:
                    positive = shifted
                delta = shifted-uv
                for j in range(len(points)):
                    vertex_rows.append(dict(frame=frame, space=space, dof=dof, sign=sign, vertex=j,
                                            x_mm=points[j, 0], y_mm=points[j, 1], z_mm=points[j, 2],
                                            z_cam_mm=depth[j], u_px=uv[j, 0], v_px=uv[j, 1],
                                            du_px=delta[j, 0], dv_px=delta[j, 1], displacement_px=float(np.linalg.norm(delta[j])),
                                            subset='near' if j in near else 'far' if j in far else 'middle'))
            overlays.append((space, dof, step, positive))
        image = np.asarray(Image.open(state['source']['frames'][frame]['path']).convert('RGB'))
        # Shared color scale for rotations within a frame; translation has its own scale.
        for space, dof, step, shifted in overlays:
            extent = max(np.linalg.norm(v-uv, axis=1).max() for _, d, _, v in overlays if d[0] == dof[0])
            make_overlay(args.output/'overlays'/f'frame-{frame:04d}_{space}_{dof}.png', image, uv, shifted,
                         depth, near, far, f'Frame {frame}: {space} {dof} +{step:g} '+('mm' if dof[0]=='T' else 'degree'), extent)
    return rows, vertex_rows


def actual_deltas(args, state, poses, points):
    deltas, contributions = [], []
    for first, second in zip(args.range[:-1], args.range[1:]):
        a, b = poses[first], poses[second]
        dt, camera_w, local_w = relative_delta(a, b)
        row = dict(frame_from=first, frame_to=second)
        for prefix, vector in [('translation', dt), ('camera_rotation', camera_w), ('local_rotation', local_w)]:
            row.update({prefix+'_'+axis:float(value) for axis, value in zip(AXES, vector)})
        row.update(translation_magnitude_mm=float(np.linalg.norm(dt)), angular_magnitude_deg=float(np.linalg.norm(local_w)))
        deltas.append(row)
        base = project(points, a, state['camera']); near, far, _ = depth_split(points, a, args.near_fraction)
        translation, rotation = split_pose(a, b)
        variants = dict(translation=translation, rotation=rotation, full=b)
        for space, vector in [('camera', camera_w), ('local', local_w)]:
            for i, component in enumerate(vector):
                variants[space+'_R'+AXES[i]] = perturb(a, 'rotation', i, component, space)
        for i, component in enumerate(dt):
            variants['camera_T'+AXES[i]] = perturb(a, 'translation', i, component)
        vectors = {name:project(points, value, state['camera'])-base for name, value in variants.items()}
        result = dict(frame_from=first, frame_to=second)
        for name, vector in vectors.items():
            result.update({name+'_'+k:v for k, v in displacement(base, base+vector, near, far).items()})
        # Contributions interact: squared RMS values are not additive.
        tr, ro = vectors['translation'], vectors['rotation']
        result['translation_rotation_cosine'] = float(np.sum(tr*ro)/(np.linalg.norm(tr)*np.linalg.norm(ro))) if np.linalg.norm(tr)*np.linalg.norm(ro)>0 else None
        result['nonlinear_composition_residual_px'] = float(np.sqrt(np.mean(np.sum((vectors['full']-tr-ro)**2, axis=1))))
        for space in ('camera', 'local'):
            axis_sum = sum(vectors[space+'_R'+axis] for axis in AXES)
            result[space+'_axis_sum_residual_px'] = float(np.sqrt(np.mean(np.sum((ro-axis_sum)**2, axis=1))))
        contributions.append(result)
    return deltas, contributions


def aggregate(deltas, contributions):
    d = {key:series_stats([row[key] for row in deltas]) for key in deltas[0] if key not in ('frame_from', 'frame_to')}
    c = {key:series_stats([row[key] for row in contributions]) for key in contributions[0]
         if key not in ('frame_from', 'frame_to') and all(row[key] is not None for row in contributions)}
    return dict(frame_pairs=len(deltas), deltas=d, contributions=c, rotation_modes={space:rotation_modes([[row[space+'_rotation_'+axis] for axis in AXES] for row in deltas]) for space in ('camera','local')})


def plot_series(path, rows, columns, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in columns:
        ax.plot([r['frame_to'] for r in rows], [r[key] for r in rows], '.-', label=label)
    ax.axhline(0, color='gray', linewidth=.5); ax.set(title=title, xlabel='Destination frame index', ylabel=ylabel)
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def ranking(rows, frame, space, kind):
    positive = [r for r in rows if r['frame']==frame and r['space']==space and r['dof'][0]==kind and r['sign']==1]
    return sorted(positive, key=lambda r:r['rms_px_per_unit'], reverse=True)


def report(args, state, rows, summary):
    a = summary['all_static']; d, c = a['deltas'], a['contributions']
    reference = args.frames[0]
    pose_rankings = {str(frame):{space:ranking(rows,frame,space,'R')[0]['dof'] for space in ('camera','local')} for frame in args.frames}
    best_t = ranking(rows, reference, 'camera', 'T')[0]
    best_c = ranking(rows, reference, 'camera', 'R')[0]
    best_l = ranking(rows, reference, 'local', 'R')[0]
    translation_ties = '/'.join(r['dof'] for r in ranking(rows, reference, 'camera', 'T') if np.isclose(r['rms_px_per_unit'],best_t['rms_px_per_unit']))
    noisy = {s:max(AXES, key=lambda axis:d[s+'_rotation_'+axis]['rms']) for s in ('camera','local')}
    real = {s:max(AXES, key=lambda axis:c[s+'_R'+axis+'_rms_px']['rms']) for s in ('camera','local')}
    ratio = c['rotation_rms_px']['rms']/c['translation_rms_px']['rms']
    dominant = 'rotation' if ratio > 1 else 'translation'
    summary['answers'] = dict(reference_frame=reference, most_sensitive_translation=best_t,
                              most_sensitive_camera_rotation=best_c, most_sensitive_local_rotation=best_l,
                              noisiest_rotation_axis=noisy, largest_actual_rotation_screen_axis=real,
                              rotation_translation_rms_ratio=ratio, largest_isolated_screen_contribution=dominant, representative_rotation_rankings=pose_rankings)
    lines = ['# Диагностика residual jitter: 370test1', '',
             f'Исходный project: `{args.project.resolve()}`. Read-only poses: `{args.track.resolve()}`.',
             f'Static {args.range[0]}–{args.range[-1]} (PNG {state["source"]["frames"][args.range[0]]["filename"]}–{state["source"]["frames"][args.range[-1]]["filename"]}); {a["frame_pairs"]} adjacent pairs.',
             f'Representative frames: {", ".join(map(str,args.frames))}. Основной sensitivity frame: {reference}.',
             'Preset e002: top_confidence; visib=0.30; PnP=2.0 crop px; iterations=5; pad=0.10; crop=280×280.',
             f'Camera: `{json.dumps(state["camera"])}`; mesh extents 80×160×11 мм. Inference не повторялся.', '',
             '## Основной вывод', '',
             f'Rotation-only {c["rotation_rms_px"]["rms"]:.3f} px и translation-only {c["translation_rms_px"]["rms"]:.3f} px сопоставимы. Full adjacent-pose RMS {c["full_rms_px"]["rms"]:.3f} px. Rotation/translation ratio {ratio:.3f}; этот ratio описывает isolated contributions, а не причинную долю jitter.',
             f'Local sensitivity: {best_l["dof"]}; largest angular variation: R{noisy["local"]}; largest actual rotational screen contribution: R{real["local"]}. Эти три ranking различаются.',
             f'Most screen-sensitive rotation по representative frames (camera/local): {pose_rankings}. Чувствительность и near/far отношения меняются с ракурсом.', '',
             '## Метод и conventions', '',
             '`T_cam_from_object`: X_cam right, Y_cam down, Z_cam forward; translation mm. Pinhole projection в исходные RGB pixels.',
             'Camera rotation: R′=exp(w_cam)R; local rotation: R′=R exp(w_local). Translation t фиксирована: вращаем вокруг object origin, а не вокруг camera origin.',
             'Relative rotations: w_local=log(R_nᵀ R_next); w_camera=log(R_next R_nᵀ); w_camera=R_n w_local.',
             'Для SO(3) log используется scipy Rotation, которая устраняет float32 отклонения ортогональности только в расчёте angular delta; сохранённые poses/проекции не корректируются.',
             '[SciPy composition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.__mul__.html), [rotation vectors](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html).',
             f'Фиксированы {summary["vertices"]} уникальных mesh vertices; при лимите берётся deterministic равномерный subset индексов. Vertex identities одинаковы во всех тестах.',
             'Vertex weighting равномерный по vertices, не по площади поверхности; у этого mesh vertices сосредоточены у углов. Используются все surface vertices, включая невидимые; без z-buffer/occlusion weighting. Точки вне RGB не отбрасываются, negative-depth запрещён.',
             f'Near/far: ближайшие/дальние {args.near_fraction*100:g}% vertices по Z_cam базовой pose; subsets фиксированы для ± perturbations одной pose, пересчитываются для другой pose.',
             '± perturbations сохранены отдельно. Таблицы ниже показывают + step; normalized ranking использует px/mm или px/degree внутри соответствующего типа.',
             'Per-axis actual contribution: exp одного компонента rotvec, остальные нули; first-order интерпретация, суммы не являются независимым разложением variance.',
             'RMS aggregate = sqrt(mean(pair_RMS²)); p95 в агрегатах — percentile по pair RMS, в sensitivity — по vertex displacement.',
             'Pair screen RMS отличается от предыдущего 0.856 px dispersion относительно одной representative pose: нельзя сравнивать их как улучшение/ухудшение. Это screen variation, не absolute accuracy. Static включает RGB differences, возможное движение камеры и начальную convergence. Ground truth отсутствует.', '',
             '## Test A/C — sensitivity и ranking', '',
             '| Frame | Space | DOF | Step | RMS px | p95 px | Max px | Near RMS | Far RMS | Near/Far | px/unit |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for frame in args.frames:
        for space, kind in [('camera','T'),('camera','R'),('local','R')]:
            for r in ranking(rows, frame, space, kind):
                lines.append(f'| {frame} | {space} | {r["dof"]} | {r["step"]} {r["unit"]} | {r["rms_px"]:.3f} | {r["p95_px"]:.3f} | {r["max_px"]:.3f} | {r["near_rms_px"]:.3f} | {r["far_rms_px"]:.3f} | {r["near_far_ratio"]:.3f} | {r["rms_px_per_unit"]:.3f} |')
    lines += ['', '## Test B — реальные angular/translation deltas', '',
              '| Component | Mean | Std | RMS | Median abs | p95 abs | Sign changes | Lag-1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, stat in d.items():
        lag = 'n/a' if stat['lag1_autocorrelation'] is None else f'{stat["lag1_autocorrelation"]:.3f}'
        sign = 'n/a' if stat['sign_change_rate'] is None else f'{stat["sign_change_rate"]:.3f}'
        lines.append(f'| {key} | {stat["mean"]:.4f} | {stat["std"]:.4f} | {stat["rms"]:.4f} | {stat["median_abs"]:.4f} | {stat["p95_abs"]:.4f} | {sign} | {lag} |')
    lines += ['', 'Angular components в degrees, translation в mm. Sign-change denominator исключает пары с нулём (counts в summary).',
              'Высокая смена знака с отрицательной lag-1 — descriptive evidence, не статистическое доказательство wobble: соседние first differences сами имеют отрицательную корреляцию даже при iid pose noise.', '',
              '## Rotational covariance modes', '',
              f'Camera variance fractions: {a["rotation_modes"]["camera"]["variance_fractions"]}.',
              f'Local variance fractions: {a["rotation_modes"]["local"]["variance_fractions"]}.',
              f'Principal local direction [X,Y,Z]: {a["rotation_modes"]["local"]["principal_axes_xyz"][0]}.',
              'PCA — angular covariance, не screen-weighted ranking; component signs у eigenvectors произвольны. Доли modes показывают, насколько вариация сосредоточена вдоль одного направления.', '',
              '## Screen contribution', '', '| Variant | Mean px | Median px | RMS px | p95 pair RMS px | Near RMS px | Far RMS px |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['translation','rotation','full']+[s+'_R'+a for s in ('camera','local') for a in AXES]+['camera_T'+a for a in AXES]:
        stat = c[name+'_rms_px']
        lines.append(f'| {name} | {stat["mean"]:.3f} | {stat["median"]:.3f} | {stat["rms"]:.3f} | {stat["p95"]:.3f} | {c[name+"_near_rms_px"]["rms"]:.3f} | {c[name+"_far_rms_px"]["rms"]:.3f} |')
    lines += ['', f'Translation/rotation projected-vector mean cosine: {c["translation_rotation_cosine"]["mean"]:.3f}. Negative означает частичную компенсацию.',
              f'Nonlinear (full − translation − rotation) RMS: {c["nonlinear_composition_residual_px"]["rms"]:.6f} px.',
              f'First-order axis-sum residual: camera {c["camera_axis_sum_residual_px"]["rms"]:.6f} px; local {c["local_axis_sum_residual_px"]["rms"]:.6f} px.',
              f'Full pose: near/far RMS ratio {c["full_near_rms_px"]["rms"]/c["full_far_rms_px"]["rms"]:.3f}; в реальных deltas большая экранная вариация у far-side. Near-depth amplification не универсален.',
              'Вклады не складываются скалярно: direction, correlations и cancellation важны; крупнейший isolated вклад не равен доле net jitter.', '',
              '## Проверка начальной convergence', '']
    if summary.get('settled_static'):
        settled = summary['settled_static']['contributions']
        lines += [f'Дополнительный диапазон {args.range[4]}–{args.range[-1]} исключает первые четыре poses, как в прошлой работе; основной range не подменяется.',
                  f'Translation/rotation/full RMS: {settled["translation_rms_px"]["rms"]:.3f} / {settled["rotation_rms_px"]["rms"]:.3f} / {settled["full_rms_px"]["rms"]:.3f} px.']
    lines += ['', '## Ответы на 10 вопросов', '',
              f'1. Translation sensitivity: camera {translation_ties}, {best_t["rms_px_per_unit"]:.3f} px/mm.',
              f'2. Camera rotation sensitivity: {best_c["dof"]}, {best_c["rms_px_per_0_1_degree"]:.3f} px/0.1°.',
              f'3. Local rotation sensitivity: {best_l["dof"]}, {best_l["rms_px_per_0_1_degree"]:.3f} px/0.1°.',
              f'4. Для dominant local DOF: near {best_l["near_rms_px"]:.3f} px, far {best_l["far_rms_px"]:.3f} px, ratio {best_l["near_far_ratio"]:.3f}; amplification зависит от оси/pose (см. таблицу).',
              f'5. Наибольший angular RMS: camera R{noisy["camera"]} ({d["camera_rotation_"+noisy["camera"]]["rms"]:.4f}°); local R{noisy["local"]} ({d["local_rotation_"+noisy["local"]]["rms"]:.4f}°).',
              f'6. Rotational sign-change rates {min((d[s+"_rotation_"+a]["sign_change_rate"] or 0) for s in ("camera","local") for a in AXES):.3f}–{max((d[s+"_rotation_"+a]["sign_change_rate"] or 0) for s in ("camera","local") for a in AXES):.3f}; отрицательный lag-1 у {sum(d[s+"_rotation_"+a]["lag1_autocorrelation"] is not None and d[s+"_rotation_"+a]["lag1_autocorrelation"]<0 for s in ("camera","local") for a in AXES)} из 6 компонентов. Alternating tendency есть, но устойчивый +/− цикл одной оси не установлен; first differences и 23 пары ограничивают вывод.',
              f'7. Isolated translation RMS {c["translation_rms_px"]["rms"]:.3f} px; rotation {c["rotation_rms_px"]["rms"]:.3f} px; full {c["full_rms_px"]["rms"]:.3f} px. Крупнейший isolated вклад: {dominant}.',
              f'8. Реальный rotational screen contribution: camera R{real["camera"]} ({c["camera_R"+real["camera"]+"_rms_px"]["rms"]:.3f} px); local R{real["local"]} ({c["local_R"+real["local"]+"_rms_px"]["rms"]:.3f} px).',
              f'9. Local: noisy R{noisy["local"]}, sensitive {best_l["dof"]}, actual screen R{real["local"]}; camera: noisy R{noisy["camera"]}, sensitive {best_c["dof"]}, actual screen R{real["camera"]}.',
              f'10. Rotation/translation isolated RMS ratio {ratio:.3f}. '+(f'Rotation isolated вклад больше на {(ratio-1)*100:.1f}%; оба вклада существенны. Убедительного вывода о преимущественно rotational jitter или одной доминирующей локальной оси нет.' if ratio>1 else 'Гипотеза о преимущественно rotational visible jitter по isolated screen contributions не подтверждается.'), '',
              '## Следующий этап camera/mesh research (только выводы)', '',
              'При отдельном исследовании следует проверять совместное conditioning rotation/translation и их экранную компенсацию, а не выбирать ось по angular RMS.',
              'Sensitivity зависит от mesh origin, геометрических lever arms, перспективы и глубины. Local X/Y/Z относятся к prepared mesh: X≈80 мм ширина, Y≈160 мм длина, Z≈11 мм толщина; это отличается от legacy prototype mesh.',
              'Без независимых image landmarks/силуэтов и ground truth нельзя различить focal/geometry mismatch, реальное движение камеры, RGB ambiguity и pose bias. Ни camera, ни mesh здесь не оптимизировались.',
              'Никакие фильтры, smoothing, corrections, production настройки/UI и poses не изменены.', '',
              '## Integrity / артефакты', '',
              '`config.json`: provenance, hashes, settings, args, definitions. `summary.json`: aggregates и answers.',
              '`tables/`: ± sensitivity, per-vertex vectors, signed pose deltas, split/per-axis screen contributions. `plots/`: все requested series.',
              '`overlays/`: 9 DOF maps каждого representative frame; цвет — true px, стрелки ×20; hidden mesh points включены.',
              '`integrity_check.json`: byte hashes inputs до/после. Этот анализ не меняет исходное tracking solution.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--track', type=Path, required=True, help='Saved research poses/<candidate> directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--range', type=parse_range, default=parse_range('2:25'))
    parser.add_argument('--frames', type=lambda x:list(map(int,x.split(','))), default=[2,42,50,54,62,70])
    parser.add_argument('--translation-mm', type=float, default=1.)
    parser.add_argument('--rotation-deg', type=float, default=.1)
    parser.add_argument('--near-fraction', type=float, default=.25)
    parser.add_argument('--max-vertices', type=int, default=2000)
    args = parser.parse_args()
    state = read(args.project/'project.json'); root = args.track.parent.parent
    config = read(root/'config.json'); candidate = read(root/'summaries'/f'{args.track.name}.json')
    validate_inputs(args, state, config, candidate)
    poses, files = load_track(args.track)
    indices = sorted(set(args.frames+args.range))
    if any(i not in poses for i in indices):
        raise ValueError('Required frame missing from saved track')
    if len({read(files[i])['mode'] for i in args.range}) != 1:
        raise ValueError('Range crosses independently initialized static/motion tracks')
    protected = {args.project/'project.json', Path(state['mesh']['prepared']), Path(state['mesh']['source']),
                 root/'config.json', root/'summaries'/f'{args.track.name}.json'} | set(files.values())
    for i in indices:
        source = Path(state['source']['frames'][i]['path'])
        if sha(source) != config['source_hashes'][str(i)]:
            raise ValueError(f'Source RGB changed at frame {i}')
        protected.add(source)
    before = {str(p.resolve()):sha(p) for p in sorted(protected)}
    points = np.unique(load_mesh(state['mesh']['prepared']).vertices_mm, axis=0)
    if len(points) > args.max_vertices:
        points = points[np.linspace(0, len(points)-1, args.max_vertices).astype(int)]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('tables','plots','overlays'):
        (args.output/name).mkdir()
    np.save(args.output/'vertices_mm.npy', points)
    metadata = dict(args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    camera=state['camera'], mesh=state['mesh'], candidate=candidate['settings'],
                    input_sha256=before, versions=dict(numpy=np.__version__, scipy=version('scipy'), matplotlib=version('matplotlib'), python=sys.version),
                    tool_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('pose_sensitivity_metrics.py')]},
                    pose_convention='T_cam_from_object; mm; OpenCV; rotation pivot = object origin; fixed t',
                    vertex_count=len(points), depth_fraction=args.near_fraction)
    write_json(args.output/'config.json', metadata)
    rows, vertices = sensitivity(args, state, poses, points)
    deltas, contributions = actual_deltas(args, state, poses, points)
    write_csv(args.output/'tables/sensitivity.csv', rows)
    write_csv(args.output/'tables/vertex_displacements.csv', vertices)
    write_csv(args.output/'tables/frame_deltas.csv', deltas)
    write_csv(args.output/'tables/screen_contributions.csv', contributions)
    summary = dict(vertices=len(points), all_static=aggregate(deltas,contributions))
    if len(args.range)>6:
        summary['settled_static'] = aggregate(deltas[4:],contributions[4:])
    for space in ('camera','local'):
        plot_series(args.output/'plots'/f'rotation_{space}.png', deltas,
                    [(space+'_rotation_'+a, 'R'+a) for a in AXES], space+' angular delta (SO(3))', 'Degrees')
    plot_series(args.output/'plots/translation.png', deltas, [('translation_'+a,'T'+a) for a in AXES], 'Camera translation delta', 'mm')
    plot_series(args.output/'plots/magnitudes.png', deltas, [('translation_magnitude_mm','Translation (mm)'),('angular_magnitude_deg','Rotation (deg)')], 'Adjacent pose magnitudes (separate units)', 'mm / degrees')
    plot_series(args.output/'plots/screen_contribution.png', contributions, [(n+'_rms_px',n) for n in ('translation','rotation','full')], 'Isolated screen displacement', 'RMS px')
    for space in ('camera','local'):
        plot_series(args.output/'plots'/f'screen_axes_{space}.png', contributions,
                    [(space+'_R'+a+'_rms_px','R'+a) for a in AXES], space+' isolated angular components', 'RMS px')
    report(args,state,rows,summary)
    write_json(args.output/'summary.json',summary)
    after = {p:sha(p) for p in before}
    unchanged = before == after
    write_json(args.output/'integrity_check.json',dict(unchanged=unchanged,before=before,after=after))
    if not unchanged:
        raise RuntimeError('Inputs changed during analysis; inspect integrity_check.json before using results')
    print(json.dumps(summary['answers'],indent=2))


if __name__ == '__main__':
    main()
