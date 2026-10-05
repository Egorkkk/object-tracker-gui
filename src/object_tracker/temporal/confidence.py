"""Generic, optional quality inputs; no tracker or UI dependencies."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FrameQuality:
    confidence: float | None = None
    correspondence_confidence: float | None = None
    inlier_ratio: float | None = None
    valid_correspondences: float | None = None
    reprojection_error: float | None = None
    failed: bool = False


DEFAULT_WEIGHTS = dict(confidence=1., correspondence_confidence=.5, inlier_ratio=1.,
                       valid_correspondences=.25, reprojection_error=.5)


def reliability(quality, weights=None):
    """Weighted mean over available inputs only; absent metrics are neutral.

    Counts saturate at 100; pixel error uses 1/(1+(error/4)^2).
    The score is a heuristic reliability, not a calibrated probability.
    """
    if quality.failed:
        return 0.
    total = value = 0.
    for name, weight in (DEFAULT_WEIGHTS if weights is None else weights).items():
        metric = getattr(quality, name)
        if metric is None or not math.isfinite(metric) or weight <= 0:
            continue
        if name == 'valid_correspondences':
            metric = metric / 100.
        elif name == 'reprojection_error':
            metric = 1. / (1. + (max(0., metric) / 4.) ** 2)
        value += weight * min(1., max(0., metric))
        total += weight
    return value / total if total else 1.


def quality_from_record(record):
    data = record.get('backend_diagnostics', {}).get('frame_quality', {})
    return FrameQuality(
        confidence=data.get('confidence', record.get('score')),
        correspondence_confidence=data.get('correspondence_confidence'),
        inlier_ratio=data.get('inlier_ratio'),
        valid_correspondences=data.get('valid_correspondences'),
        reprojection_error=data.get('reprojection_error'),
        failed=record.get('status') == 'FAILED' or data.get('failed', False))
