from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import convert_obj_to_glb, generate_mesh, segment_and_crop

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="TrueRender Demo")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
active_job_id: str | None = None


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        jobs[job_id].update(updates)


def _run_job(job_id: str, image_path: Path, prompt: str, job_dir: Path) -> None:
    global active_job_id

    try:
        _set_job(job_id, status="running", stage="segmenting")
        canonical_png = segment_and_crop(image_path, prompt, job_dir)

        _set_job(job_id, stage="cropping", preview_url=f"/outputs/{job_id}/canonical/canonical_rgba.png")

        _set_job(job_id, stage="generating mesh")
        obj_path = generate_mesh(canonical_png, job_dir)

        _set_job(job_id, stage="converting preview")
        obj_download = job_dir / "mesh.obj"
        if obj_path.resolve() != obj_download.resolve():
            shutil.copy2(obj_path, obj_download)
        glb_path = convert_obj_to_glb(obj_download, job_dir / "mesh.glb")

        _set_job(
            job_id,
            status="done",
            stage="done",
            mesh_url=f"/outputs/{job_id}/{glb_path.name}",
            preview_url=f"/outputs/{job_id}/{glb_path.name}",
            obj_url=f"/outputs/{job_id}/{obj_download.name}",
            glb_url=f"/outputs/{job_id}/{glb_path.name}",
            error=None,
        )
    except Exception as exc:
        traceback.print_exc()
        _set_job(job_id, status="error", stage="error", error=str(exc))
    finally:
        with jobs_lock:
            if active_job_id == job_id:
                active_job_id = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/reconstruct/image")
async def reconstruct_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form("the main object"),
) -> dict[str, str]:
    global active_job_id

    with jobs_lock:
        if active_job_id is not None:
            raise HTTPException(status_code=429, detail="A reconstruction job is already running. Try again shortly.")

        job_id = str(uuid.uuid4())
        active_job_id = job_id
        jobs[job_id] = {
            "status": "pending",
            "stage": "queued",
            "mesh_url": None,
            "preview_url": None,
            "obj_url": None,
            "glb_url": None,
            "error": None,
        }

    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.png").suffix or ".png"
    image_path = job_dir / f"input{suffix}"

    with image_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    background_tasks.add_task(_run_job, job_id, image_path, prompt or "the main object", job_dir)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)
