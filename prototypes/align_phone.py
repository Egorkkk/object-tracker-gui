import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


FX = 1867.0
FY = 1867.0
CX = 960.0
CY = 540.0


def euler_to_matrix(rx, ry, rz):
    """Euler degrees -> R. Order: Rz @ Ry @ Rx."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def make_pose(rx, ry, rz, tx, ty, tz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_to_matrix(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def project_points(vertices, T):
    R = T[:3, :3]
    t = T[:3, 3]

    pts = (R @ vertices.T).T + t
    valid = pts[:, 2] > 1.0

    pts = pts[valid]
    if len(pts) == 0:
        return np.empty((0, 2)), valid

    uv = np.empty((len(pts), 2), dtype=np.float64)
    uv[:, 0] = FX * pts[:, 0] / pts[:, 2] + CX
    uv[:, 1] = FY * pts[:, 1] / pts[:, 2] + CY

    return uv, valid


def project_xyz_axes(T, axis_len=100.0):
    pts = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ], dtype=np.float64)

    R = T[:3, :3]
    t = T[:3, 3]
    p = (R @ pts.T).T + t

    uv = np.zeros((4, 2), dtype=np.float64)

    for i in range(4):
        if p[i, 2] <= 1:
            return None

        uv[i, 0] = FX * p[i, 0] / p[i, 2] + CX
        uv[i, 1] = FY * p[i, 1] / p[i, 2] + CY

    return np.round(uv).astype(np.int32)


def render_overlay(image, vertices, T):
    out = image.copy()

    uv, valid = project_points(vertices, T)

    h, w = image.shape[:2]

    if len(uv) >= 3:
        # Не даём огромным выбросам мешать convexHull.
        reasonable = (
            (uv[:, 0] > -2 * w)
            & (uv[:, 0] < 3 * w)
            & (uv[:, 1] > -2 * h)
            & (uv[:, 1] < 3 * h)
        )

        uv2 = uv[reasonable]

        if len(uv2) >= 3:
            hull = cv2.convexHull(
                np.round(uv2).astype(np.int32)
            )

            cv2.polylines(
                out,
                [hull],
                True,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Показываем выборку projected vertices.
        if len(uv2) > 0:
            step = max(1, len(uv2) // 2500)

            for p in uv2[::step]:
                x, y = np.round(p).astype(int)
                if 0 <= x < w and 0 <= y < h:
                    out[y, x] = (0, 180, 255)

    axes = project_xyz_axes(T)

    if axes is not None:
        o, x, y, z = axes

        # OpenCV BGR:
        cv2.line(out, tuple(o), tuple(x), (0, 0, 255), 3, cv2.LINE_AA)   # X
        cv2.line(out, tuple(o), tuple(y), (0, 255, 0), 3, cv2.LINE_AA)   # Y
        cv2.line(out, tuple(o), tuple(z), (255, 0, 0), 3, cv2.LINE_AA)   # Z

        cv2.putText(out, "X", tuple(x), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.putText(out, "Y", tuple(y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(out, "Z", tuple(z), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

    return out


def save_pose(T, overlay, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "initial_pose.npy", T)
    np.savetxt(
        out_dir / "initial_pose.txt",
        T,
        fmt="%.9f"
    )
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    print("\nSaved:")
    print(out_dir / "initial_pose.npy")
    print(out_dir / "initial_pose.txt")
    print(out_dir / "overlay.png")
    print("\nT_cam_from_model:")
    print(T)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", default="custom/frame000.png")
    ap.add_argument("--mesh", default="custom/phone_mm.ply")
    ap.add_argument("--out", default="custom/alignment")

    ap.add_argument("--rx", type=float, default=90.0)
    ap.add_argument("--ry", type=float, default=0.0)
    ap.add_argument("--rz", type=float, default=0.0)

    ap.add_argument("--tx", type=float, default=0.0)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=700.0)

    ap.add_argument("--no-gui", action="store_true")

    args = ap.parse_args()

    image = cv2.imread(args.image)

    if image is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    print("Image:", image.shape[1], "x", image.shape[0])

    mesh = trimesh.load(args.mesh)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    print("Mesh:")
    print("  vertices:", len(mesh.vertices))
    print("  faces:", len(mesh.faces))
    print("  extents mm:", mesh.extents)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    # Для UI полные 175k вершин не нужны.
    if len(vertices) > 30000:
        ids = np.linspace(
            0,
            len(vertices) - 1,
            30000,
            dtype=int
        )
        vertices = vertices[ids]

    out_dir = Path(args.out)

    if args.no_gui:
        T = make_pose(
            args.rx, args.ry, args.rz,
            args.tx, args.ty, args.tz
        )

        overlay = render_overlay(image, vertices, T)
        save_pose(T, overlay, out_dir)
        return

    # ----------------------------------------
    # Interactive mode
    # ----------------------------------------

    cv2.namedWindow("GoTrack initial alignment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GoTrack initial alignment", 1280, 720)

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 700, 350)

    # rotation:
    # 0..3600 => -180..+180 degrees
    cv2.createTrackbar(
        "RX",
        "Controls",
        int((args.rx + 180) * 10),
        3600,
        lambda x: None
    )
    cv2.createTrackbar(
        "RY",
        "Controls",
        int((args.ry + 180) * 10),
        3600,
        lambda x: None
    )
    cv2.createTrackbar(
        "RZ",
        "Controls",
        int((args.rz + 180) * 10),
        3600,
        lambda x: None
    )

    # translation X/Y:
    # 0..2000 => -1000..+1000 mm
    cv2.createTrackbar(
        "TX mm",
        "Controls",
        int(args.tx + 1000),
        2000,
        lambda x: None
    )
    cv2.createTrackbar(
        "TY mm",
        "Controls",
        int(args.ty + 1000),
        2000,
        lambda x: None
    )

    # Z = 50..3000 mm
    cv2.createTrackbar(
        "TZ mm",
        "Controls",
        int(args.tz),
        3000,
        lambda x: None
    )

    print()
    print("Controls:")
    print("  sliders : pose")
    print("  S       : save pose + overlay")
    print("  P       : print matrix")
    print("  Q / ESC : quit")
    print()

    last_T = None
    last_overlay = None

    while True:
        rx = cv2.getTrackbarPos("RX", "Controls") / 10.0 - 180.0
        ry = cv2.getTrackbarPos("RY", "Controls") / 10.0 - 180.0
        rz = cv2.getTrackbarPos("RZ", "Controls") / 10.0 - 180.0

        tx = cv2.getTrackbarPos("TX mm", "Controls") - 1000.0
        ty = cv2.getTrackbarPos("TY mm", "Controls") - 1000.0

        tz = max(
            50.0,
            float(cv2.getTrackbarPos("TZ mm", "Controls"))
        )

        T = make_pose(rx, ry, rz, tx, ty, tz)
        overlay = render_overlay(image, vertices, T)

        text = (
            f"R=({rx:.1f}, {ry:.1f}, {rz:.1f}) deg   "
            f"T=({tx:.0f}, {ty:.0f}, {tz:.0f}) mm"
        )

        cv2.putText(
            overlay,
            text,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("GoTrack initial alignment", overlay)

        last_T = T
        last_overlay = overlay

        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            break

        elif key == ord("p"):
            print("\nT_cam_from_model:")
            print(T)

        elif key == ord("s"):
            save_pose(T, overlay, out_dir)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
