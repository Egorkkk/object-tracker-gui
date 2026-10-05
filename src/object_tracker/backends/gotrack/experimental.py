"""Opt-in correspondence selection and process-local research hooks.

Research hooks run in a dedicated lab process; the sampling-only context also
supports the single-owner UI worker. Original functions/options are restored
on exit; upstream/runtime files are not edited.
"""
from contextlib import contextmanager
import inspect
import textwrap
import numpy as np


def select_correspondences(points, confidence, max_count, strategy="random", grid=8):
    points = np.asarray(points)
    confidence = np.asarray(confidence)
    if points.shape != (len(confidence), 2) or confidence.ndim != 1:
        raise ValueError("Expected Nx2 points and N confidence values")
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("Correspondences must be finite")
    if max_count < 1 or grid < 1:
        raise ValueError("Counts must be positive")
    if strategy not in ("random", "top_confidence", "spatial_confidence"):
        raise ValueError("Unknown correspondence strategy")
    count = min(len(points), max_count)
    if strategy == "random":
        # Match upstream, including random permutation at exactly max_count.
        return (np.random.choice(len(points), count, replace=False)
                if len(points) >= max_count else np.arange(len(points)))
    order = np.argsort(-confidence, kind="stable")
    if strategy == "top_confidence" or not count:
        return order[:count]
    # A grid over target points' bounding box, not frame resolution. Round-robin
    # ranks within occupied cells prevents one high-confidence region dominating.
    extent = np.maximum(np.ptp(points, axis=0), 1e-9)
    cells = np.minimum(((points - points.min(axis=0)) / extent * grid).astype(int), grid-1)
    labels = cells[:, 1] * grid + cells[:, 0]
    ranks = np.empty(len(points), dtype=int)
    seen = np.zeros(grid * grid, dtype=int)
    for idx in order:
        cell = labels[idx]
        ranks[idx] = seen[cell]
        seen[cell] += 1
    return np.lexsort((np.arange(len(points)), -confidence, ranks))[:count]


def plain(value):
    # Hydra leaves some nested opts as OmegaConf containers.
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "_asdict"):
        return {k: plain(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def actual_settings(model):
    from utils.config import PnPOpts
    settings = plain(model.opts)
    settings['pnp_opts'] = plain(model.opts.pnp_opts or PnPOpts())
    settings['correspondence'] = 'random'
    return settings


def _selection_function(original_pnp, strategy):
    if strategy == 'random':
        return original_pnp
    if strategy not in ('top_confidence', 'spatial_confidence'):
        raise ValueError('Unknown correspondence strategy')
    # Change only the sampling boundary; keep solver/evaluation/fallback intact.
    code = textwrap.dedent(inspect.getsource(original_pnp))
    needle = '''sampled_ids = np.random.choice(
                        len(corresp_2d_),
                        pnp_opts.max_num_corresps,
                        replace=False,
                    )'''
    if code.count(needle) != 1:
        raise RuntimeError('Upstream PnP changed: review experimental hook')
    replacement = "sampled_ids = _lab_select(corresp_2d_, corresp_weight_, pnp_opts.max_num_corresps)"
    namespace = dict(original_pnp.__globals__)
    namespace['_lab_select'] = lambda p, w, n: select_correspondences(p, w, n, strategy)
    exec(compile(code.replace(needle, replacement), '<jitter-lab-pnp>', 'exec'), namespace)
    return namespace['poses_from_correspondences']


@contextmanager
def correspondence_selection(strategy):
    """Scoped sampling-only hook for the single-owner GoTrack worker."""
    if strategy == 'random':
        yield
        return
    from utils import pnp_util
    original = pnp_util.poses_from_correspondences
    pnp_util.poses_from_correspondences = _selection_function(original, strategy)
    try:
        yield
    finally:
        pnp_util.poses_from_correspondences = original


@contextmanager
def experiment(backend, settings, trace):
    """Apply only explicit deltas to actual baseline; capture PnP and iterations."""
    from utils import pnp_util
    from utils.config import PnPOpts
    model = backend.model
    original_opts = model.opts
    original_pnp = pnp_util.poses_from_correspondences
    original_forward = model.forward_pipeline
    strategy = settings.get('correspondence', 'random')
    opts = {k: v for k, v in settings.items() if k in original_opts._fields and k != 'pnp_opts'}
    if 'crop_size' in opts:
        opts['crop_size'] = tuple(opts['crop_size'])
    if settings.get('pnp_opts'):
        opts['pnp_opts'] = (original_opts.pnp_opts or PnPOpts())._replace(**settings['pnp_opts'])
    model.opts = original_opts._replace(**opts)
    try:
        function = _selection_function(original_pnp, strategy)

        def observed_pnp(*args, **kwargs):
            bound = inspect.signature(original_pnp).bind(*args, **kwargs)
            bound.apply_defaults()
            weights = bound.arguments['corresps_weight'].detach().cpu().numpy()
            threshold = bound.arguments['weight_threshold']
            count = np.sum(weights > threshold, axis=(1, 2))
            result = function(*args, **kwargs)
            # Upstream marks too-few-points as failed=False and can append two
            # diagnostic entries on RANSAC failure. Do not rely on failed alone.
            trace.setdefault('pnp', []).append(dict(
                available=count.tolist(), selected=np.minimum(count, (model.opts.pnp_opts or PnPOpts()).max_num_corresps).tolist(),
                failed=np.asarray(result['failed']).tolist(),
                quality=np.asarray(result['quality']).tolist(),
                too_few=bool(np.any(count < 6))))
            return result

        def observed_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            trace['iterations'] = [outputs[f'iter={i}_pred_poses_orig_cam_from_model'][0].detach().cpu().numpy().tolist()
                                   for i in range(model.opts.num_iterations_test)]
            return outputs

        pnp_util.poses_from_correspondences = observed_pnp
        model.forward_pipeline = observed_forward
        yield
    finally:
        model.opts = original_opts
        pnp_util.poses_from_correspondences = original_pnp
        model.forward_pipeline = original_forward
