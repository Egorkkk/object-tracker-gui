"""CPU-only PnP telemetry regression; run via scripts/run_python.sh."""
from pathlib import Path
import numpy as np
import torch
from object_tracker.backends.gotrack.environment import activate_runtime, prepare_runtime
from object_tracker.backends.gotrack.experimental import correspondence_selection
from object_tracker.backends.gotrack.quality import capture_quality
root = Path(__file__).resolve().parents[1]
activate_runtime(prepare_runtime(root/'external/gotrack', root/'outputs/runtime'))
from utils import pnp_util
from utils.config import PnPOpts
rng=np.random.default_rng(5)
points=rng.uniform([-40,-40,-40],[40,40,40],(1,10,10,3)).astype(np.float32)
m=np.eye(4,dtype=np.float32);m[2,3]=700
k=np.array([[60,0,5],[0,60,5],[0,0,1]],np.float32)
cam=points+np.array([0,0,700]);uv=cam@k.T;uv=uv[...,:2]/uv[...,2:]
weights=np.linspace(.4,.95,100).reshape(1,10,10).astype(np.float32)
kwargs=dict(corresps_2d=torch.tensor(uv),corresps_3d=torch.tensor(points),corresps_weight=torch.tensor(weights),masks=torch.ones(1,10,10),intrinsics=torch.tensor(k[None]),pnp_solver_name='opencv',init_poses=torch.tensor(m.T.copy()).T[None],pnp_opts=PnPOpts(max_num_corresps=40))
original=pnp_util.poses_from_correspondences
original_eval=pnp_util.eval_pnp_output
for strategy in ['random','top_confidence']:
 with correspondence_selection(strategy):
  np.random.seed(17);raw=pnp_util.poses_from_correspondences(**kwargs)
 with correspondence_selection(strategy),capture_quality() as trace:
  np.random.seed(17);observed=pnp_util.poses_from_correspondences(**kwargs)
 for key in raw:
  if key == 'run_time': continue
  if isinstance(raw[key],torch.Tensor):torch.testing.assert_close(raw[key],observed[key],rtol=0,atol=0)
  else:np.testing.assert_array_equal(raw[key],observed[key])
 assert trace[0]['valid_correspondences']==100
 assert trace[0]['selected_correspondences']==40
 assert 'reprojection_error' in trace[0]
 assert trace[0]['num_inliers']==100
 assert pnp_util.poses_from_correspondences is original
 assert pnp_util.eval_pnp_output is original_eval
 print(strategy,trace)
# Existing upstream reports too-few correspondences as failed=False. Capture
# the real weak-measurement condition without changing upstream's result.
kwargs['corresps_weight']=torch.zeros_like(kwargs['corresps_weight'])
with capture_quality() as trace:
 result=pnp_util.poses_from_correspondences(**kwargs)
assert trace[0]['failed'] and trace[0]['confidence']==0
print('too-few-points:',trace)
