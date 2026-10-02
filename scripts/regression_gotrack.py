"""Execute unchanged references and the adapter in independent GPU processes."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker(args):
    from object_tracker.backends.gotrack.environment import prepare_runtime, activate_runtime, backend_version
    runtime = prepare_runtime(ROOT / "external/gotrack", args.output / "runtime")
    activate_runtime(runtime)
    import cv2
    import torch
    import random
    from hydra.utils import instantiate
    from utils import net_util, renderer_builder, structs
    import model.gotrack
    checkpoint = ROOT / "external/gotrack/gotrack_checkpoint.pt"
    link = runtime / "gotrack_checkpoint.pt"
    if not link.exists():
        link.symlink_to(checkpoint)
    out = args.output / args.worker
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps({"backend_version": backend_version(ROOT / "external/gotrack"), "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__, "gpu": torch.cuda.get_device_name(0), "camera": [1867., 1867., 960., 540.], "runtime_source_hash": (runtime / ".ready").read_text()}, indent=2))
    data = args.data
    frames = sorted((data / "frames").glob("*.png"))[:args.frames]
    if not frames:
        raise ValueError("No regression frames")
    os.chdir(runtime)
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    cv2.setRNGSeed(12345)
    if args.worker.startswith("prototype"):
        single = args.worker.endswith("refine")
        path = ROOT / "prototypes" / ("refine_one.py" if single else "track_sequence.py")
        spec = importlib.util.spec_from_file_location("reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if single:
            sys.argv = [str(path), "--image", str(frames[0]), "--mesh", str(data / "phone_mm.ply"),
                        "--pose", str(data / "initial_pose.npy"), "--out", str(out)]
        else:
            module.FRAMES_GLOB = str(data / "frames/*.png")
            module.MESH_PATH = str(data / "phone_mm.ply")
            module.INITIAL_POSE_PATH = str(data / "initial_pose.npy")
            module.OUT_DIR = out
            module.MAX_FRAMES = args.frames
        module.main()
    else:
        from object_tracker.backends.gotrack.backend import GoTrackBackend
        from object_tracker.core.mesh import load_mesh
        from object_tracker.core.types import CameraIntrinsics, Pose, TrackingRange
        backend = GoTrackBackend(ROOT / "external/gotrack", checkpoint, out / "internal", args.output / "runtime")
        mesh = load_mesh(data / "phone_mm.ply")
        initial = Pose(np.load(data / "initial_pose.npy"))
        h, w = cv2.imread(str(frames[0])).shape[:2]
        camera = CameraIntrinsics(w, h, 1867., 1867., 960., 540.)
        def image(index):
            return cv2.cvtColor(cv2.imread(str(frames[index])), cv2.COLOR_BGR2RGB)
        try:
            if args.worker.endswith("refine"):
                result = backend.refine_frame(image(0), mesh, camera, initial)
                np.save(out / "refined_pose.npy", result.pose.T_cam_from_object)
            else:
                (out / "poses").mkdir(exist_ok=True)
                scores = []
                for result in backend.track_range(image, mesh, camera, initial, TrackingRange(0, len(frames)-1)):
                    np.save(out / "poses" / f"{result.frame_index:04d}.npy", result.pose.T_cam_from_object)
                    scores.append(result.score)
                (out / "scores.json").write_text(json.dumps(scores))
        finally:
            backend.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "testdata/phone_short")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/regression")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--worker", choices=["prototype_refine", "adapter_refine", "prototype_sequence", "adapter_sequence"])
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if args.output == args.data or args.data in args.output.parents:
        raise ValueError("Regression output must be separate from source data")
    args.output.mkdir(parents=True, exist_ok=True)
    for stage in (() if args.compare_only else ("prototype_refine", "adapter_refine", "prototype_sequence", "adapter_sequence")):
        print(f"Running {stage}", flush=True)
        with (args.output / f"{stage}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", stage,
                            "--data", str(args.data), "--output", str(args.output),
                            "--frames", str(args.frames)], stdout=log, stderr=subprocess.STDOUT, check=True)
    differences = []
    pairs = [("refine", args.output / "prototype_refine/refined_pose.npy", args.output / "adapter_refine/refined_pose.npy")]
    pairs += [(p.stem, p, args.output / "adapter_sequence/poses" / p.name)
              for p in sorted((args.output / "prototype_sequence/poses").glob("*.npy"))]
    if len(pairs) != min(args.frames, len(list((args.data / "frames").glob("*.png")))) + 1:
        raise RuntimeError("Missing regression frame results")
    for name, left, right in pairs:
        a, b = np.load(left).astype(float), np.load(right).astype(float)
        # Relative rotation via solve avoids apparent nonzero self-angle due to float32 R drift.
        relative = b[:3, :3] @ np.linalg.inv(a[:3, :3])
        dr = float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))))
        dt = float(np.linalg.norm(a[:3, 3]-b[:3, 3]))
        differences.append(dict(frame=name, translation_mm=dt, rotation_deg=dr,
                                max_matrix_difference=float(np.max(np.abs(a-b)))))
    report = {"frames": len(pairs)-1, "seed": 12345, "differences": differences,
              "inputs": {str(p.relative_to(args.data)): sha256(p) for p in
                         [args.data / "phone_mm.ply", args.data / "initial_pose.npy"] +
                         sorted((args.data / "frames").glob("*.png"))[:args.frames]},
              "checkpoint_sha256": sha256(ROOT / "external/gotrack/gotrack_checkpoint.pt")}
    reference_scores = [float(row["score"]) for row in csv.DictReader(
        (args.output / "prototype_sequence/tracking.csv").open())]
    adapter_scores = json.loads((args.output / "adapter_sequence/scores.json").read_text())
    report["max_score_difference"] = float(np.max(np.abs(np.array(reference_scores)-adapter_scores)))
    refine_score = float(next(csv.DictReader((args.output / "prototype_refine/per_frame_refined_poses_000000.json").open()))["score"])
    adapter_score = float(next(csv.DictReader((args.output / "adapter_refine/internal/per_frame_refined_poses_000000.json").open()))["score"])
    report["refine_score_difference"] = abs(refine_score - adapter_score)
    report["passed"] = report["refine_score_difference"] < 0.001 and all(d["translation_mm"] < 0.1 and d["rotation_deg"] < 0.1 for d in differences) and report["max_score_difference"] < 0.001
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=2))
    if not report["passed"]:
        raise SystemExit("Regression differences require investigation")


if __name__ == "__main__":
    main()
