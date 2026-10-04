#!/usr/bin/env python3
"""Standalone staged GoTrack benchmark. Run with scripts/run_python.sh.

Project JSON and all input assets are read-only. Each run uses a new directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from object_tracker.backends.gotrack.backend import GoTrackBackend
from object_tracker.backends.gotrack.experimental import actual_settings, experiment
from object_tracker.core.mesh import load_mesh
from object_tracker.core.project import Project, file_identity
from object_tracker.core.types import CameraIntrinsics, Pose
from jitter_metrics import dispersion, stats, project

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed % (2**31-1))


def frame_range(text):
    start, end = map(int, text.split(':'))
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError('Expected inclusive start:end')
    return list(range(start, end+1))


def preview(image, points, pose, camera, label, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    uv = project(points, pose, camera)
    uv = uv[np.isfinite(uv).all(axis=1)]
    uv = uv[(np.abs(uv[:, 0]) < camera.width*3) & (np.abs(uv[:, 1]) < camera.height*3)]
    if len(uv) >= 3:
        cv2.polylines(image, [cv2.convexHull(np.round(uv).astype(np.int32))], True, (0,255,0), 2)
    cv2.putText(image, label, (20,35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def evaluate(backend, p, mesh, camera, points, initial, settings, exp_id, args):
    renderer=getattr(backend.model,'renderer',None)
    if renderer is not None and renderer.renderer is not None:
        if tuple(renderer.im_size)!=tuple(settings['actual']['crop_size']):
            backend._release_renderer()
    folder = args.output / 'poses' / exp_id
    folder.mkdir(parents=True, exist_ok=True)
    result = dict(id=exp_id, settings=settings, modes={}, timestamp=datetime.now(timezone.utc).isoformat())
    for mode, indices in [('repeat', [args.frame]*args.count), ('static', args.static), ('motion', args.motion)]:
        current = initial
        records = []
        # Motion starts at the last static pose for continuity; the explicit
        # motion initial pose is persisted and varies only through refinement.
        if mode == 'motion' and result['modes'].get('static', {}).get('last_pose'):
            current = Pose(np.array(result['modes']['static']['last_pose']))
        mode_initial = current.T_cam_from_object.tolist()
        seed_all(args.seed)
        for sample, index in enumerate(indices):
            run_seed = args.seed if settings.get('reset_seed') else args.seed+sample
            seed_all(run_seed)
            trace = {}
            record = dict(frame=index, sample=sample, seed=run_seed, mode=mode, experiment=exp_id)
            image = p.sequence.rgb(index).copy()
            start = time.perf_counter()
            try:
                with experiment(backend, settings['overrides'], trace):
                    refined = backend.refine_frame(image, mesh, camera, initial if mode == 'repeat' else current, index)
                matrix = refined.pose.T_cam_from_object
                previous = current.T_cam_from_object
                dt = float(np.linalg.norm(matrix[:3,3]-previous[:3,3]))
                from scipy.spatial.transform import Rotation
                dr = float(np.rad2deg(Rotation.from_matrix(previous[:3,:3].T @ matrix[:3,:3]).magnitude()))
                pnp_failure = any(t['too_few'] or any(t['failed']) or any(q <= 0 for q in t['quality']) for t in trace.get('pnp',[]))
                warn = refined.score < args.min_score or (sample > 0 and (dt > args.jump_mm or dr > args.jump_deg))
                record.update(matrix=matrix.tolist(), score=refined.score, runtime=refined.runtime,
                              delta_mm=dt if sample else None, delta_deg=dr if sample else None,
                              pnp_failure=pnp_failure, failure=refined.score <= 0, warning=bool(warn), trace=trace)
                current = refined.pose
                if mode != 'repeat':
                    preview(image, points, matrix, camera, f'{exp_id} frame={index} score={refined.score:.3f}',
                            args.output/'previews'/exp_id/mode/f'{index:06d}.jpg')
            except Exception as exc:
                record.update(failure=True, warning=True, pnp_failure=True, error=f'{type(exc).__name__}: {exc}',
                              runtime=time.perf_counter()-start, trace=trace)
            records.append(record)
            save(folder/f'{mode}-{sample:04d}.json',record)
            print(f'{exp_id} {mode} {sample+1}/{len(indices)} score={record.get("score")} error={record.get("error", "")}',flush=True)
            # Architecture failures are recorded once rather than repeated.
            if record.get('error') and settings['overrides'].get('crop_size'):
                break
        successful = [r for r in records if not r['failure'] and 'matrix' in r]
        metrics = dict(requested=len(indices), attempted=len(records), successes=len(successful),
                       failures=sum(r['failure'] for r in records), warning_count=sum(r['warning'] for r in records),
                       pnp_failures=sum(r['pnp_failure'] for r in records),
                       pnp_failed_iterations=sum(sum(t['too_few'] or any(t['failed']) or any(q <= 0 for q in t['quality']) for t in r.get('trace',{}).get('pnp',[])) for r in records),
                       completed=len(records)==len(indices) and len(successful)==len(indices),
                       failure_rate=1-len(successful)/len(indices), initial_pose=mode_initial,
                       score_mean=float(np.mean([r['score'] for r in successful])) if successful else None,
                       score_min=float(np.min([r['score'] for r in successful])) if successful else None,
                       runtime=stats([r['runtime'] for r in records]),
                       large_translation_jumps=sum((r.get('delta_mm') or 0)>args.jump_mm for r in records),
                       large_rotation_jumps=sum((r.get('delta_deg') or 0)>args.jump_deg for r in records),
                       last_pose=current.T_cam_from_object.tolist(), errors=[r['error'] for r in records if 'error' in r])
        if successful and mode != 'motion':
            metrics['dispersion'], series = dispersion([r['matrix'] for r in successful], points, camera)
            for r, t, a, s in zip(successful, series['translation'], series['rotation'], series['screen']):
                r.update(translation_mm=float(t), rotation_deg=float(a), screen_px=float(s) if np.isfinite(s) else None)
            # Update durable records with per-frame dispersion values.
            for r in successful:
                save(folder/f'{mode}-{r["sample"]:04d}.json',r)
        result['modes'][mode] = metrics
    save(args.output/'summaries'/f'{exp_id}.json',result)
    return result


def ranking(result):
    s, m = result['modes']['static'], result['modes']['motion']
    jitter = s.get('dispersion',{}).get('screen',{}).get('rms')
    return (not m['completed'], m['failures'], m['large_translation_jumps']+m['large_rotation_jumps'],
            m['pnp_failures'], jitter if jitter is not None else float('inf'))


def report(results, args, config):
    rows=[]
    for r in results:
        settings=r['settings']['actual']
        for mode, m in r['modes'].items():
            disp=m.get('dispersion',{})
            rows.append(dict(experiment=r['id'], stage=r['settings']['stage'], parent=r['settings']['parent'], mode=mode,
                selector=settings['correspondence'],visib=settings['visib_threshold'],pnp_px=settings['pnp_opts']['pnp_inlier_thresh'],
                iters=settings['num_iterations_test'],pad=settings['crop_rel_pad'],crop=settings['crop_size'][0],
                t_mm=disp.get('translation',{}).get('rms'),r_deg=disp.get('rotation',{}).get('rms'),screen_px=disp.get('screen',{}).get('rms'),
                score=m['score_mean'],min_score=m['score_min'],failures=m['failures'],pnp_failures=m['pnp_failures'],warnings=m['warning_count'],
                runtime_s=m['runtime']['median'],completed=m['completed']))
    with (args.output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    selectable=[r for r in results if not r['settings']['reset_seed']]
    save(args.output/'summary.json',dict(experiments=results, recommended=min(selectable,key=ranking)['id']))
    lines=['# GoTrack jitter lab', '', 'Все poses сырые; camera/mesh фиксированы; production defaults не изменены.',
           'Motion не имеет ground truth: score и отсутствие ошибок не доказывают точность.',
           f'Проект: {config["project"]}; lens metadata: {args.lens_mm} mm. Intrinsics прочитаны из проекта.',
           f'Кадр repeat: {args.frame}; count: {args.count}; static: {args.static}; motion: {args.motion}.',
           'Индексы zero-based, диапазоны включительные. Повторы используют одну initial pose; sequence — propagation.',
           'Статистика: median translation, mean SO(3) rotation; screen RMS по фиксированным mesh points без clipping.',
           'Std angular distance — std величины углов; дополнительно доступны RMS, median, MAD, p95.',
           'Seeds NumPy/Torch/OpenCV задаются перед каждым кадром. CUDA deterministic algorithms не включены.',
           '', '## Actual baseline', '```json',json.dumps(config['baseline'],indent=2), '```', '',
           '| Experiment | Mode | Selector | visib | PnP px | Iters | Pad | Crop | T RMS mm | R RMS deg | Screen RMS px | Score | Fail / PnP |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return '—' if x is None else f'{x:.4f}'
    for row in rows:
        lines.append(f'| {row["experiment"]} | {row["mode"]} | {row["selector"]} | {row["visib"]} | {row["pnp_px"]} | {row["iters"]} | {row["pad"]} | {row["crop"]} | {fmt(row["t_mm"])} | {fmt(row["r_deg"])} | {fmt(row["screen_px"])} | {fmt(row["score"])} | {row["failures"]} / {row["pnp_failures"]} |')
    lines+=['', '## Lineage and limitations']
    for r in results:
        lines.append(f'- {r["id"]}: stage={r["settings"]["stage"]}, parent={r["settings"]["parent"]}, overrides={r["settings"]["overrides"]}; errors={r["modes"]["motion"]["errors"] or r["modes"]["repeat"]["errors"]}')
    lines+=['','Automatic ranking uses completion, failures, jumps, PnP failures, then static screen jitter. Visual inspection is required before recommending a preset.']
    (args.output/'report.md').write_text('\n'.join(lines))
    plots(results,args)


def plots(results,args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=args.output/'plots';folder.mkdir(exist_ok=True)
    best=min([r for r in results if not r['settings']['reset_seed']],key=ranking)
    chosen=[results[0]]+([] if best is results[0] else [best])
    for mode in ['repeat','static']:
        fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
        for r in chosen:
            records=[json.loads(p.read_text()) for p in sorted((args.output/'poses'/r['id']).glob(f'{mode}-*.json'))]
            for ax,key in zip(axes,['translation_mm','rotation_deg','screen_px','score']):
                ax.plot([v['sample'] if mode=='repeat' else v['frame'] for v in records],[v.get(key,np.nan) for v in records],label=r['id']);ax.set_ylabel(key);ax.grid(True);ax.legend()
        axes[-1].set_xlabel('repeat' if mode=='repeat' else 'frame index');fig.tight_layout();fig.savefig(folder/f'{mode}.png');plt.close(fig)
    for stage,key in [('visib','visib_threshold'),('iterations','num_iterations_test')]:
        stage_file=args.output/f'stage-{stage}.json'
        ids=set(json.loads(stage_file.read_text())['candidates']) if stage_file.exists() else set()
        subset=[r for r in results if r['id'] in ids or r['settings']['stage']==stage]
        if not subset:continue
        fig,ax=plt.subplots(figsize=(8,4))
        for parent in sorted(set(r['settings']['actual']['correspondence'] for r in subset)):
            group=sorted([r for r in subset if r['settings']['actual']['correspondence']==parent],key=lambda r:r['settings']['actual'][key])
            ax.plot([r['settings']['actual'][key] for r in group],[r['modes']['static'].get('dispersion',{}).get('screen',{}).get('rms',np.nan) for r in group],'o-',label=parent)
        ax.set_xlabel(key);ax.set_ylabel('Static screen RMS px');ax.grid(True);ax.legend();fig.tight_layout();fig.savefig(folder/f'{stage}.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Continue a saved run with matching inputs and options')
    parser.add_argument('--retry-failed',action='store_true',help='Archive and retry experiments with recorded errors on resume')
    parser.add_argument('--frame',type=int,default=2)
    parser.add_argument('--static',type=frame_range,default=frame_range('2:25'))
    parser.add_argument('--motion',type=frame_range,default=frame_range('26:70'))
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--lens-mm',type=float,default=40,help='Metadata only; never changes camera')
    parser.add_argument('--stages',default='selection,visib,pnp,iterations,pad,crop')
    parser.add_argument('--survivors',type=int,default=1)
    parser.add_argument('--min-score',type=float,default=.1)
    parser.add_argument('--jump-mm',type=float,default=50)
    parser.add_argument('--jump-deg',type=float,default=30)
    parser.add_argument('--gotrack',type=Path,default=ROOT/'external/gotrack')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'external/gotrack/gotrack_checkpoint.pt')
    args=parser.parse_args()
    if args.count<20:parser.error('Repeatability requires at least 20 runs')
    if args.survivors not in (1,2):parser.error('survivors must be 1 or 2')
    if args.output.exists() and not args.resume:parser.error('Output must be a new directory, or use --resume')
    state=json.loads(args.project.read_text() if args.project.name=='project.json' else (args.project/'project.json').read_text())
    directory=args.project.parent if args.project.name=='project.json' else args.project
    p=Project(directory,state)  # Never open/save/recover the production project.
    for idx in [args.frame]+args.static+args.motion:p.check_index(idx)
    if args.static[0]!=args.frame:parser.error('static must start at frame with initial pose')
    if args.motion[0]!=args.static[-1]+1:parser.error('motion must follow static to preserve propagation')
    initial=p.pose(args.frame)
    mesh=load_mesh(state['mesh']['prepared']);camera=CameraIntrinsics(**state['camera'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    logging.basicConfig(filename=args.output/'inference.log',level=logging.INFO)
    backend=GoTrackBackend(args.gotrack,args.checkpoint,args.output/'internal',ROOT/'outputs/runtime')
    try:
        backend.initialize()
        baseline=actual_settings(backend.model)
        config=dict(project=str(args.project.resolve()),project_snapshot=state,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    git_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
                    gotrack_commit=backend.version,backend_metadata=backend.metadata,baseline=baseline,
                    camera=state['camera'],mesh=state['mesh'],mesh_sha256=file_identity(mesh.source_path),
                    initial_pose=initial.T_cam_from_object.tolist(),timestamp=datetime.now(timezone.utc).isoformat(),
                    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    source_hashes={str(i):file_identity(p.sequence.frames[i]['path']) for i in sorted(set([args.frame]+args.static+args.motion))},
                    lab_sources={str(f.relative_to(ROOT)):file_identity(f) for f in [Path(__file__),ROOT/'tools/jitter_metrics.py',ROOT/'src/object_tracker/backends/gotrack/experimental.py']})
        if args.resume:
            old=json.loads((args.output/'config.json').read_text())
            for key in ['baseline','camera','mesh_sha256','initial_pose','source_hashes','gotrack_commit']:
                if old[key]!=config[key]:parser.error(f'Resume input mismatch: {key}')
            for key in ['frame','static','motion','count','seed','stages','survivors','min_score','jump_mm','jump_deg']:
                if old['args'][key]!=config['args'][key]:parser.error(f'Resume option mismatch: {key}')
            save(args.output/f'resume-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}.json',config)
        else:
            save(args.output/'config.json',config)
        for filename,digest in config['lab_sources'].items():
            snapshot=args.output/'source_snapshots'/digest/filename
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes((ROOT/filename).read_bytes())
        points=mesh.vertices_mm[np.linspace(0,len(mesh.vertices_mm)-1,min(1000,len(mesh.vertices_mm)),dtype=int)]
        np.save(args.output/'screen_points_mm.npy',points)
        results=[]
        def run(exp_id,stage,parent,overrides,reset=False):
            actual=json.loads(json.dumps(baseline));actual.update({k:v for k,v in overrides.items() if k!='pnp_opts'})
            actual['pnp_opts'].update(overrides.get('pnp_opts',{}))
            setting=dict(stage=stage,parent=parent,overrides=overrides,actual=actual,reset_seed=reset)
            saved=args.output/'summaries'/f'{exp_id}.json'
            if args.resume and args.retry_failed and saved.exists():
                previous=json.loads(saved.read_text())
                if any(m['errors'] for m in previous['modes'].values()):
                    archive=args.output/'failed_attempts'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')/exp_id
                    archive.mkdir(parents=True)
                    shutil.move(saved,archive/'summary.json')
                    for kind in ['poses','previews']:
                        path=args.output/kind/exp_id
                        if path.exists():shutil.move(path,archive/kind)
            if args.resume and saved.exists():
                r=json.loads(saved.read_text())
                previous_settings={k:v for k,v in r['settings'].items() if k!='source_hashes'}
                if previous_settings!=setting:parser.error(f'Resume experiment mismatch: {exp_id}')
                print(f'Reusing completed {exp_id}',flush=True)
            else:
                setting['source_hashes']=config['lab_sources']
                r=evaluate(backend,p,mesh,camera,points,initial,setting,exp_id,args)
            results.append(r);report(results,args,config);return r
        base=run('baseline','repeatability',None,{})
        run('baseline_fixed_seed','determinism','baseline',{},True)
        parents=[base]
        grids=dict(selection=('correspondence',['random','top_confidence','spatial_confidence']),
                   visib=('visib_threshold',sorted(set([.3,.4,.5,.6,baseline['visib_threshold']]))),
                   pnp=('pnp_inlier_thresh',sorted(set([1.,1.5,2.,baseline['pnp_opts']['pnp_inlier_thresh']]))),
                   iterations=('num_iterations_test',sorted(set([1,2,3,5,baseline['num_iterations_test']]))),
                   pad=('crop_rel_pad',sorted(set([max(0,baseline['crop_rel_pad']-.05),baseline['crop_rel_pad'],baseline['crop_rel_pad']+.05,baseline['crop_rel_pad']+.1]))),
                   crop=('crop_size',[280,336,420]))
        for stage in filter(None,args.stages.split(',')):
            key,values=grids[stage];candidates=[]
            for parent in parents:
                for value in values:
                    overrides=json.loads(json.dumps(parent['settings']['overrides']))
                    if stage=='pnp':overrides.setdefault('pnp_opts',{})[key]=value
                    else:overrides[key]=[value,value] if stage=='crop' else value
                    actual=json.loads(json.dumps(baseline));actual.update({k:v for k,v in overrides.items() if k!='pnp_opts'})
                    actual['pnp_opts'].update(overrides.get('pnp_opts',{}))
                    equivalent=next((r for r in results if r['settings']['actual']==actual and not r['settings']['reset_seed']),None)
                    if equivalent:candidates.append(equivalent);continue
                    candidates.append(run(f'e{len(results):03d}',stage,parent['id'],overrides))
            eligible=sorted(candidates,key=ranking)
            parents=[base]+[r for r in eligible if r is not base][:args.survivors]
            save(args.output/f'stage-{stage}.json',dict(parents=[r['id'] for r in parents],candidates=[r['id'] for r in candidates]))
        report(results,args,config)
    finally:
        backend.shutdown()


if __name__=='__main__':main()
