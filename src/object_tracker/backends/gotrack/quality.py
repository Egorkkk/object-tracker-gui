"""Read-only scalar telemetry around existing PnP; solver inputs stay untouched.

GoTrack exposes weighted inlier quality but discards its supporting scalars.
Hooks are scoped to the application's single inference worker and restored even
on failure. No RNG calls, new inference, or image analysis.
"""
from contextlib import contextmanager
import inspect
import numpy as np


@contextmanager
def capture_quality():
    from utils import pnp_util
    from utils.config import PnPOpts
    original = pnp_util.poses_from_correspondences
    original_eval = pnp_util.eval_pnp_output
    trace = []
    evaluations = []

    def observed_eval(*args, **kwargs):
        bound = inspect.signature(original_eval).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        result = original_eval(*args, **kwargs)
        # The existing evaluator already computes these residuals but only
        # returns an RGB error tile. Preserve a scalar in crop-camera pixels.
        import cv2
        points = a['corresp_3d'] @ cv2.Rodrigues(a['rvec_est_m2c'])[0].T + np.asarray(a['t_est_m2c']).reshape(1, 3)
        projected = points @ a['intrinsic'].T
        projected = projected[:, :2] / projected[:, 2:3]
        projected[:, 0] = np.clip(projected[:, 0], 0, a['width']-1)
        projected[:, 1] = np.clip(projected[:, 1], 0, a['height']-1)
        error = float(np.median(np.linalg.norm(a['corresp_2d']-projected, axis=1)))
        evaluations.append(dict(num_inliers=int(result['num_inliers']), inlier_ratio=float(result['num_inliers']) / max(1, len(points)),
                                reprojection_error=error if np.isfinite(error) else None))
        return result

    def observed(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        weights = a['corresps_weight'].detach().cpu().numpy()
        valid = weights > a['weight_threshold']
        values = weights[valid]
        count = int(valid.sum())
        evaluations.clear()
        result = original(*args, **kwargs)
        opts = a['pnp_opts'] or PnPOpts()
        entry = dict(valid_correspondences=count, selected_correspondences=min(count, opts.max_num_corresps),
                     correspondence_confidence=float(values.mean()) if len(values) else 0.,
                     failed=count < 6 or bool(np.any(result['failed'])),
                     confidence=float(np.asarray(result['quality']).reshape(-1)[-1]))
        if evaluations:
            entry.update(evaluations[-1])
        if entry['failed']:
            entry['confidence'] = 0.
        trace.append(entry)
        return result

    # Top-confidence selection uses a copied function namespace. Capture its
    # evaluator too, without touching the selection algorithm.
    evaluator_globals = original.__globals__
    previous_eval = evaluator_globals.get('eval_pnp_output')
    evaluator_globals['eval_pnp_output'] = observed_eval
    pnp_util.poses_from_correspondences = observed
    pnp_util.eval_pnp_output = observed_eval
    try:
        yield trace
    finally:
        pnp_util.poses_from_correspondences = original
        pnp_util.eval_pnp_output = original_eval
        if previous_eval is not None:
            evaluator_globals['eval_pnp_output'] = previous_eval
        else:
            evaluator_globals.pop('eval_pnp_output', None)
