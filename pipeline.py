from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

SAM_PREDICTOR = None
TRIPOSR_REPO = Path(os.environ.get("TRIPOSR_REPO", "/content/TripoSR"))
MODELS_READY = False


def reset_dir(path: Path | str) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _patch_triposr() -> None:
    # Restore files before patching so this cell is idempotent even after a failed previous patch.
    subprocess.run(["git", "checkout", "--", "run.py", "tsr/utils.py"], check=True, cwd=TRIPOSR_REPO)

    # Patch TripoSR so --no-remove-bg does not import rembg/onnxruntime at module load time.
    # We already provide a SAM-masked object composited onto gray, so rembg is unnecessary.
    run_py = Path(TRIPOSR_REPO) / "run.py"
    code = run_py.read_text()
    code = code.replace(
        "import rembg\n",
        "# TRUERENDER_PATCH_LAZY_REMBG_RUN: rembg is imported lazily only when needed\nrembg = None\n",
        1,
    )
    expected = "if args.no_remove_bg:\n    rembg_session = None\nelse:\n    rembg_session = rembg.new_session()"
    patched = "if args.no_remove_bg:\n    rembg_session = None\nelse:\n    import rembg\n    rembg_session = rembg.new_session()"
    if expected not in code:
        raise RuntimeError("Could not find expected rembg_session block in TripoSR run.py")
    code = code.replace(expected, patched, 1)

    expected = "out_mesh_path = os.path.join(output_dir, str(i), f\"mesh.{args.model_save_format}\")"
    patched = "out_mesh_path = os.path.join(output_dir, str(i), f\"mesh.{args.model_save_format}\")\nos.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)"
    if expected not in code:
        raise RuntimeError("Could not find expected out_mesh_path line in TripoSR run.py")
    code = code.replace(expected, patched, 1)

    run_py.write_text(code)
    print("Patched TripoSR run.py for lazy rembg import and robust export directory creation")

    utils_py = Path(TRIPOSR_REPO) / "tsr" / "utils.py"
    code = utils_py.read_text()
    code = code.replace(
        "import rembg\n",
        "# TRUERENDER_PATCH_LAZY_REMBG_UTILS: rembg is imported lazily only when remove_background is called\nrembg = None\n",
        1,
    )
    expected = "    if do_remove:\n        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)"
    patched = "    if do_remove:\n        import rembg\n        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)"
    if expected not in code:
        raise RuntimeError("Could not find expected remove_background block in TripoSR tsr/utils.py")
    code = code.replace(expected, patched, 1)
    utils_py.write_text(code)
    print("Patched TripoSR tsr/utils.py to avoid rembg import when unused")


def load_models() -> None:
    """Load SAM 3 lazily and verify the patched TripoSR runtime."""
    global SAM_PREDICTOR, MODELS_READY

    if MODELS_READY and SAM_PREDICTOR is not None:
        return

    if not TRIPOSR_REPO.exists():
        raise RuntimeError(
            f"TripoSR repo not found at {TRIPOSR_REPO}. Run setup_colab.ipynb before launching the app."
        )

    if str(TRIPOSR_REPO) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_REPO))

    _patch_triposr()

    probe = subprocess.run(
        [sys.executable, "-c", "import sys; import tsr.utils; print(sys.executable); print('tsr.utils import ok')"],
        capture_output=True,
        text=True,
        check=True,
        cwd=TRIPOSR_REPO,
    )
    print("Child Python probe:\n" + probe.stdout)

    from sam3.model_builder import build_sam3_video_predictor

    SAM_PREDICTOR = build_sam3_video_predictor()
    MODELS_READY = True
    print("TrueRender models loaded: SAM 3 predictor ready; patched TripoSR runtime verified")


def segment_and_crop(image_path: Path | str, prompt: str, output_dir: Path | str) -> Path:
    """Run SAM 3 on one image and return the canonical 768x768 RGBA crop."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    load_models()

    output_dir = Path(output_dir)
    sam_input = output_dir / "sam_input"
    masks_dir = output_dir / "masks"
    frames_rgba = output_dir / "frames_rgba"
    canonical_dir = output_dir / "canonical"
    reset_dir(sam_input)
    reset_dir(masks_dir)
    reset_dir(frames_rgba)
    reset_dir(canonical_dir)

    frame_path = sam_input / "0.jpg"
    Image.open(image_path).convert("RGB").save(frame_path, quality=95)

    try:
        resp = SAM_PREDICTOR.handle_request(dict(type="start_session", resource_path=str(sam_input)))
        sid = resp["session_id"]
        SAM_PREDICTOR.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))

        results = []
        for r in SAM_PREDICTOR.propagate_in_video(sid, propagation_direction="forward"):
            masks = r["outputs"]["out_binary_masks"]
            if len(masks) > 0:
                results.append((r["frame_index"], masks))
        SAM_PREDICTOR.close_session(sid)

        if not results:
            raise ValueError(
                f"SAM 3 found no objects matching prompt '{prompt}'. "
                "Try a more specific description (e.g. 'cup', 'shoe', 'person')."
            )

        for idx, masks in sorted(results, key=lambda x: x[0]):
            src_name = "0.jpg"
            stem = Path(src_name).stem
            rgb = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            mask = np.asarray(masks[0]).astype(bool)
            mask_u8 = (mask.astype(np.uint8) * 255)
            kernel = np.ones((3, 3), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            rgba = np.dstack([rgb, mask_u8])
            Image.fromarray(mask_u8).save(masks_dir / f"{stem}.png")
            Image.fromarray(rgba).save(frames_rgba / f"{stem}.png")

        print(f"Saved SAM outputs for {len(results)} frames")
    except Exception:
        try:
            if "sid" in locals():
                SAM_PREDICTOR.close_session(sid)
        finally:
            raise

    FRAMES_RGBA = str(frames_rgba)
    MASKS_DIR = str(masks_dir)
    CANONICAL_DIR = str(canonical_dir)

    reset_dir(CANONICAL_DIR)

    CANONICAL_SIZE = 768
    MIN_MASK_AREA = 0.015
    MAX_MASK_AREA = 0.70

    scores = []
    for mask_path in sorted(glob.glob(f"{MASKS_DIR}/*.png")):
        stem = Path(mask_path).stem
        rgba_path = f"{FRAMES_RGBA}/{stem}.png"
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        h, w = mask.shape
        area_frac = mask.mean()
        if not (MIN_MASK_AREA <= area_frac <= MAX_MASK_AREA):
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bbox_w, bbox_h = x1 - x0 + 1, y1 - y0 + 1
        bbox_area_frac = (bbox_w * bbox_h) / (w * h)
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        center_penalty = abs(cx - 0.5) + abs(cy - 0.5)
        fill_ratio = mask.sum() / max(bbox_w * bbox_h, 1)
        size_score = -abs(bbox_area_frac - 0.25)  # prefer visible object, not tiny/not clipped
        score = 2.0 * fill_ratio + 1.5 * size_score - 1.0 * center_penalty
        scores.append((score, stem, dict(area_frac=area_frac, bbox_area_frac=bbox_area_frac, fill_ratio=fill_ratio, center_penalty=center_penalty)))

    assert scores, "No usable canonical frame candidates. Inspect SAM masks."
    scores = sorted(scores, reverse=True)
    best_score, best_stem, best_meta = scores[0]
    print("Best canonical frame:", best_stem, "score=", round(best_score, 3), best_meta)

    rgba = Image.open(f"{FRAMES_RGBA}/{best_stem}.png").convert("RGBA")
    mask = np.array(Image.open(f"{MASKS_DIR}/{best_stem}.png").convert("L")) > 127
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Add padding around the object, then square-crop.
    pad = int(0.18 * max(x1 - x0 + 1, y1 - y0 + 1))
    x0, x1 = max(0, x0 - pad), min(rgba.width - 1, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(rgba.height - 1, y1 + pad)
    side = max(x1 - x0 + 1, y1 - y0 + 1)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, min(rgba.width - side, cx - side // 2))
    y0 = max(0, min(rgba.height - side, cy - side // 2))
    x1, y1 = x0 + side, y0 + side

    crop = rgba.crop((x0, y0, x1, y1))
    crop = ImageOps.contain(crop, (CANONICAL_SIZE, CANONICAL_SIZE), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (CANONICAL_SIZE, CANONICAL_SIZE), (255, 255, 255, 0))
    canvas.alpha_composite(crop, ((CANONICAL_SIZE - crop.width) // 2, (CANONICAL_SIZE - crop.height) // 2))

    canonical_png = f"{CANONICAL_DIR}/canonical_rgba.png"
    canvas.save(canonical_png)
    print("Saved canonical input:", canonical_png)
    return Path(canonical_png)


def generate_mesh(canonical_png_path: Path | str, output_dir: Path | str) -> Path:
    """Run TripoSR and return the generated OBJ path."""
    import numpy as np
    from PIL import Image

    load_models()

    output_dir = Path(output_dir)
    triposr_output = output_dir / "triposr"
    reset_dir(triposr_output)
    (triposr_output / "0").mkdir(parents=True, exist_ok=True)

    # Run TripoSR as the selected clean-mesh generator
    # TripoSR works best when --no-remove-bg receives an RGB image with a neutral gray background.
    # We already have a SAM alpha mask, so do NOT ask rembg to segment it again.
    PYTHON = sys.executable

    os.chdir(TRIPOSR_REPO)

    # Confirm we are using the same interpreter that installed TripoSR.
    probe = subprocess.run(
        [PYTHON, "-c", "import sys; print(sys.executable); print('child python ok')"],
        capture_output=True,
        text=True,
        check=True,
    )
    print("Child Python probe:\n" + probe.stdout)

    rgba = Image.open(canonical_png_path).convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4]
    gray = np.full_like(rgb, 0.5)
    composited = rgb * alpha + gray * (1.0 - alpha)
    triposr_input = f"{output_dir}/triposr_gray_bg.png"
    Image.fromarray((composited * 255).astype(np.uint8)).save(triposr_input)
    print("Saved TripoSR input:", triposr_input)

    triposr_cmd = [
        PYTHON, "-u", "run.py",
        triposr_input,
        "--output-dir", str(triposr_output),
        "--no-remove-bg",
        "--foreground-ratio", "0.85",
        # TripoSR's OBJ export is the stable path. Generate OBJ first, then convert/package after validation.
        "--model-save-format", "obj",
        "--mc-resolution", "256",
    ]
    print("Running:", " ".join(triposr_cmd))

    log_path = str(output_dir / "triposr_run.log")
    last_lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(triposr_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            last_lines.append(line)
            if len(last_lines) > 160:
                last_lines.pop(0)
        ret = proc.wait()

    if ret != 0:
        print("\n--- Last 160 lines from TripoSR run log ---")
        print("".join(last_lines))
        print(f"Full log saved at: {log_path}")
        raise subprocess.CalledProcessError(ret, triposr_cmd)

    triposr_outputs = sorted(glob.glob(f"{triposr_output}/**/*", recursive=True))
    print("TripoSR outputs:")
    for p in triposr_outputs:
        if os.path.isfile(p):
            print(" ", p, f"{os.path.getsize(p)/1e6:.1f} MB")

    mesh_path = triposr_output / "0" / "mesh.obj"
    if not mesh_path.exists():
        candidates = sorted(triposr_output.glob("**/*.obj"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise RuntimeError(f"TripoSR finished but no OBJ was found in {triposr_output}")
        mesh_path = candidates[0]
    return mesh_path


def convert_obj_to_glb(obj_path: Path | str, glb_path: Path | str) -> Path:
    import trimesh

    scene_or_mesh = trimesh.load(obj_path, force="scene")
    scene_or_mesh.export(glb_path)
    return Path(glb_path)
