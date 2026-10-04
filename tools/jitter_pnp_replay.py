"""Isolate NumPy correspondence sampling and OpenCV RNG on frozen PnP inputs."""
import argparse
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import cv2
import numpy as np

from object_tracker.backends.gotrack.backend import GoTrackBackend
from object_tracker.core.mesh import load_mesh
from object_tracker.core.types import CameraIntrinsics, Pose
from object_tracker.core.project import Project
from jitter_metrics import dispersion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--from-frozen',action='store_true',help='Replay saved NPZ without loading the GPU model again')
    args=parser.parse_args()
    root=args.run
    config=json.loads((root/'config.json').read_text())
    output=root/'pnp_replay';output.mkdir(exist_ok=True)
    project=Project(Path(config['project']).parent,config['project_snapshot'])
    camera=CameraIntrinsics(**config['camera'])
    mesh=load_mesh(config['mesh']['prepared'])
    backend=GoTrackBackend(config['args']['gotrack'],config['args']['checkpoint'],output/'internal',root/'runtime')
    import torch
    if args.from_frozen:
        from object_tracker.backends.gotrack.environment import prepare_runtime, activate_runtime
        activate_runtime(prepare_runtime(backend.source,root/'runtime'))
    else:
        backend.initialize()
    from utils import pnp_util
    seed=config['args']['seed']
    np.random.seed(seed);cv2.setRNGSeed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    original=pnp_util.poses_from_correspondences
    frozen={}
    def capture(*positional,**keywords):
        if not frozen:
            bound=inspect.signature(original).bind(*positional,**keywords);bound.apply_defaults()
            frozen.update({k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in bound.arguments.items()})
        return original(*positional,**keywords)
    if args.from_frozen:
        frozen=json.loads((output/'frozen_metadata.json').read_text())
        arrays=np.load(output/'frozen_inputs.npz')
        frozen.update({k:torch.from_numpy(arrays[k].copy()) for k in arrays.files})
    else:
        try:
            pnp_util.poses_from_correspondences=capture
            backend.refine_frame(project.sequence.rgb(config['args']['frame']).copy(),mesh,camera,Pose(np.array(config['initial_pose'])))
        finally:
            pnp_util.poses_from_correspondences=original
            backend.shutdown()
        np.savez_compressed(output/'frozen_inputs.npz',**{k:v.numpy() for k,v in frozen.items() if isinstance(v,torch.Tensor)})
        (output/'frozen_metadata.json').write_text(json.dumps({k:v for k,v in frozen.items() if not isinstance(v,torch.Tensor)},indent=2))
    # GPU->CPU copies of sliced init tvec produce contiguous arrays; a CPU-only
    # slice otherwise retains a stride OpenCV rejects as an output buffer.
    # Reproduce the GPU path's memory layout, leaving solver/sampling unchanged.
    namespace=dict(original.__globals__)
    namespace['torch_helpers']=SimpleNamespace(tensor_to_array=lambda t:np.ascontiguousarray(t.detach().cpu().numpy()))
    exec(compile(inspect.getsource(original),'<jitter-lab-cpu-pnp>','exec'),namespace)
    cpu_pnp=namespace['poses_from_correspondences']
    results={}
    points=np.load(root/'screen_points_mm.npy')
    # Crop poses use the same mm object frame; dispersion projection uses the
    # frozen crop camera so screen jitter here is explicitly in crop pixels.
    K=frozen['intrinsics'][0].numpy();h,w=frozen['masks'].shape[-2:]
    crop_camera=CameraIntrinsics(w,h,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
    for label,vary_np,vary_cv in [('fixed',False,False),('numpy_only',True,False),('opencv_only',False,True),('both',True,True)]:
        matrices=[];runs=[]
        for i in range(args.count):
            seed=config['args']['seed'];np.random.seed(seed+i if vary_np else seed);cv2.setRNGSeed(seed+i if vary_cv else seed)
            start=time.perf_counter()
            result=cpu_pnp(**{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in frozen.items()})
            matrices.append(result['estimated_poses'][0].cpu().numpy())
            runs.append(dict(matrix=matrices[-1].tolist(),quality=np.asarray(result['quality']).tolist(),runtime=time.perf_counter()-start))
        metrics,_=dispersion(matrices,points,crop_camera)
        metrics['identical_matrices']=np.array_equal(np.asarray(matrices),np.repeat(np.asarray(matrices[:1]),args.count,axis=0))
        metrics['runs']=runs
        results[label]=metrics
        print(label,{k:metrics[k] for k in ['translation_std_mm','rotation','screen','identical_matrices']},flush=True)
    (output/'summary.json').write_text(json.dumps(dict(count=args.count,units='mm / degrees / crop pixels',groups=results),indent=2))


if __name__=='__main__':main()
