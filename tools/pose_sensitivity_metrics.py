"""Read-only pinhole/SO(3) diagnostics for T_cam_from_object (mm, OpenCV)."""
import numpy as np
from scipy.spatial.transform import Rotation


def perturb(matrix, kind, axis, amount, space='camera'):
    """Rotate around the object origin, holding camera translation fixed.

    Camera: exp(w) @ R. Local: R @ exp(w). Translation uses camera axes.
    No Euler subtraction and no orbit about the camera origin.
    """
    out = np.array(matrix, dtype=float, copy=True)
    if kind == 'translation':
        if space != 'camera':
            raise ValueError('Translation perturbations use camera axes')
        out[axis, 3] += amount
    elif kind == 'rotation':
        vector = np.zeros(3); vector[axis] = np.deg2rad(amount)
        delta = Rotation.from_rotvec(vector).as_matrix()
        if space == 'camera':
            out[:3, :3] = delta @ out[:3, :3]
        elif space == 'local':
            out[:3, :3] = out[:3, :3] @ delta
        else:
            raise ValueError('Unknown coordinate space')
    else:
        raise ValueError('Unknown perturbation kind')
    return out


def relative_delta(first, second):
    r0, r1 = Rotation.from_matrix(first[:3, :3]), Rotation.from_matrix(second[:3, :3])
    return (second[:3, 3] - first[:3, 3],
            (r1 * r0.inv()).as_rotvec(degrees=True),
            (r0.inv() * r1).as_rotvec(degrees=True))


def split_pose(first, second):
    translation, rotation = np.array(first, copy=True), np.array(first, copy=True)
    translation[:3, 3] = second[:3, 3]
    rotation[:3, :3] = second[:3, :3]
    return translation, rotation


def project(points, matrix, camera):
    xyz = np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError('Fixed diagnostic vertices must all have positive depth')
    return xyz[:, :2] / xyz[:, 2:3] * [camera['fx'], camera['fy']] + [camera['cx'], camera['cy']]


def depth_split(points, matrix, fraction=.25):
    if not 0 < fraction <= .5 or len(points) < 2:
        raise ValueError('Need >=2 vertices and depth fraction in (0, .5]')
    depth = (np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
    order = np.argsort(depth, kind='stable')
    count = max(1, int(len(points) * fraction))
    return order[:count], order[-count:], depth


def displacement(reference, projected, near=None, far=None):
    distances = np.linalg.norm(np.asarray(projected) - reference, axis=1)
    result = dict(mean_px=float(distances.mean()), rms_px=float(np.sqrt(np.mean(distances**2))),
                  p95_px=float(np.percentile(distances, 95)), max_px=float(distances.max()))
    if near is not None:
        result['near_rms_px'] = float(np.sqrt(np.mean(distances[near]**2)))
        result['far_rms_px'] = float(np.sqrt(np.mean(distances[far]**2)))
        result['near_far_ratio'] = (result['near_rms_px']/result['far_rms_px']
                                   if result['far_rms_px'] > 0 else None)
    return result


def series_stats(values):
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite series')
    nonzero = np.sign(x); valid = (nonzero[1:] != 0) & (nonzero[:-1] != 0)
    lag = (float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2
           and np.std(x[:-1]) > 1e-12 and np.std(x[1:]) > 1e-12 else None)
    return dict(mean=float(x.mean()), std=float(x.std()), rms=float(np.sqrt(np.mean(x**2))),
                median=float(np.median(x)), median_abs=float(np.median(np.abs(x))),
                p95=float(np.percentile(x, 95)), p95_abs=float(np.percentile(np.abs(x), 95)),
                sign_change_rate=float(np.mean(nonzero[1:][valid] != nonzero[:-1][valid])) if valid.any() else None,
                sign_change_pairs=int(valid.sum()), lag1_autocorrelation=lag)


def rotation_modes(vectors):
    """Covariance principal axes; describes variation, not screen contribution."""
    values = np.asarray(vectors, dtype=float)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(values)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    for i in range(3):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, i])), i] < 0:
            eigenvectors[:, i] *= -1
    return dict(covariance_deg2=covariance.tolist(), eigenvalues_deg2=eigenvalues.tolist(),
                variance_fractions=(eigenvalues/eigenvalues.sum()).tolist() if eigenvalues.sum()>0 else [0.,0.,0.],
                principal_axes_xyz=eigenvectors.T.tolist())
