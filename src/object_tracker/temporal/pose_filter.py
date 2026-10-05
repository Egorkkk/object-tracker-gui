"""Offline local-linear Gaussian filtering of translation and SO(3) tangents.

Windows use actual frame indices, split at missing frames. Rotations are never
averaged as matrices or Euler angles. All output arrays are newly allocated.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .confidence import FrameQuality, reliability


@dataclass(frozen=True)
class TemporalParameters:
    enabled: bool = False
    strength: float = .6
    translation: float = 1.
    rotation: float = 1.
    adaptive: bool = True
    use_confidence: bool = True
    translation_speed: float = 100.  # mm/s
    angular_speed: float = 90.       # degrees/s
    start_frame: int | None = None
    end_frame: int | None = None

    def __post_init__(self):
        for name in ('enabled', 'adaptive', 'use_confidence'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name in ('strength', 'translation', 'rotation'):
            v = getattr(self, name)
            if isinstance(v, bool) or not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f'{name} must be between 0 and 1')
        for name in ('translation_speed', 'angular_speed'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('start_frame', 'end_frame'):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValueError(f'{name} must be a nonnegative integer')
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError('Both range endpoints are required')
        if self.start_frame is not None and self.start_frame > self.end_frame:
            raise ValueError('Range start must precede end')


def continuous_quaternions(quaternions):
    q = np.array(quaternions, dtype=float, copy=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] *= -1
    return q


def local_fit(values, offsets, weights):
    """Intercept of a weighted linear fit; reproduces constant velocity at edges."""
    total = weights.sum()
    if total < 1e-12:
        return values[np.argmin(abs(offsets))].copy()
    s1, s2 = np.dot(weights, offsets), np.dot(weights, offsets**2)
    det = total * s2 - s1*s1
    if det < 1e-10:
        return np.sum(values * weights[:, None], axis=0) / total
    coefficients = weights * (s2 - s1 * offsets) / det
    return coefficients @ values


def filter_poses(poses, parameters=None, quality=None, fps=24.):
    """poses: {frame_index: 4x4}; returns independent poses and diagnostics.

    Low-confidence frames get weak data weights and broad windows, independently
    of motion. A reliability-interpolated pilot prevents bad measurements from
    being interpreted as deliberate fast motion.
    """
    p = parameters or TemporalParameters()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive')
    from object_tracker.core.types import Pose
    indices = np.array(sorted(poses), dtype=int)
    output, diagnostics = {}, {}
    if not len(indices):
        return output, diagnostics
    matrices = np.array([Pose(poses[int(i)]).T_cam_from_object for i in indices], dtype=float)
    confidence = np.array([reliability((quality or {}).get(int(i), FrameQuality())) for i in indices])
    for segment in np.split(np.arange(len(indices)), np.flatnonzero(np.diff(indices) != 1) + 1):
        frames = indices[segment]
        raw = matrices[segment]
        c = confidence[segment]
        translations = raw[:, :3, 3]
        rotations = Rotation.from_quat(continuous_quaternions(Rotation.from_matrix(raw[:, :3, :3]).as_quat()))
        pilot_t = translations.copy()
        pilot_r = rotations
        trusted = np.flatnonzero(c >= .35) if p.use_confidence else np.arange(len(frames))
        if len(trusted):
            pilot_t = np.stack([np.interp(frames, frames[trusted], translations[trusted, axis]) for axis in range(3)], axis=1)
            if len(trusted) > 1:
                pilot_r = Slerp(frames[trusted], rotations[trusted])(np.clip(frames, frames[trusted[0]], frames[trusted[-1]]))
            else:
                pilot_r = Rotation.from_quat(np.tile(rotations[trusted[0]].as_quat(), (len(frames), 1)))
        if len(frames) > 1:
            velocity = np.linalg.norm(np.gradient(pilot_t, axis=0), axis=1) * fps
            edge_angular = (pilot_r[:-1].inv() * pilot_r[1:]).magnitude() * fps * 180 / np.pi
            angular = np.maximum(np.r_[edge_angular[0], edge_angular], np.r_[edge_angular, edge_angular[-1]])
        else:
            velocity = angular = np.zeros(1)
        for j, frame in enumerate(frames):
            result = raw[j].copy()
            in_range = p.start_frame is None or p.start_frame <= frame <= p.end_frame
            if p.enabled and p.strength > 0 and in_range:
                base = .25 + 3. * p.strength
                sigmas = []
                for amount, speed, threshold in ((p.translation, velocity[j], p.translation_speed),
                                                  (p.rotation, angular[j], p.angular_speed)):
                    factor = 1. / (1. + (speed / threshold)**2) if p.adaptive else 1.
                    if p.use_confidence:
                        factor = max(factor, 1. - c[j])
                    sigmas.append(base * amount * factor)
                radius = max(1, int(np.ceil(3 * max(sigmas))))
                window = np.arange(max(0, j-radius), min(len(frames), j+radius+1))
                offsets = frames[window] - frame
                blend = 1.
                if p.start_frame is not None:
                    # Exact raw endpoints, smooth transition over three frames.
                    edge = min(frame-p.start_frame, p.end_frame-frame) / 3.
                    u = np.clip(edge, 0., 1.)
                    blend = u*u*(3-2*u)
                for axis, sigma in enumerate(sigmas):
                    if sigma < .05:
                        continue
                    weights = np.exp(-.5 * (offsets / sigma)**2)
                    if p.use_confidence:
                        weights *= np.maximum(c[window], .001)**2
                    if axis == 0:
                        target = local_fit(translations[window], offsets, weights)
                        result[:3, 3] = translations[j] + blend * (target-translations[j])
                    else:
                        reference = pilot_r[j]
                        tangents = (reference.inv() * rotations[window]).as_rotvec()
                        target = reference * Rotation.from_rotvec(local_fit(tangents, offsets, weights))
                        delta = (rotations[j].inv() * target).as_rotvec()
                        result[:3, :3] = (rotations[j] * Rotation.from_rotvec(blend * delta)).as_matrix()
            output[int(frame)] = result
            diagnostics[int(frame)] = dict(confidence=float(c[j]), translation_velocity=float(velocity[j]),
                angular_velocity=float(angular[j]), translation_correction=float(np.linalg.norm(result[:3, 3]-translations[j])),
                rotation_correction=float((rotations[j].inv()*Rotation.from_matrix(result[:3, :3])).magnitude()*180/np.pi))
    return output, diagnostics
