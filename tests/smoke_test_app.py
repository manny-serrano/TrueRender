from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src import app as truerender_app


def test_reconstruct_image_smoke(monkeypatch):
    def fake_segment_and_crop(image_path, prompt, output_dir):
        canonical = output_dir / "canonical" / "canonical_rgba.png"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(canonical)
        return canonical

    def fake_generate_mesh(canonical_png, output_dir):
        obj = output_dir / "triposr" / "0" / "mesh.obj"
        obj.parent.mkdir(parents=True, exist_ok=True)
        obj.write_text("o smoke\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        return obj

    def fake_convert_obj_to_glb(obj_path, glb_path):
        glb_path.write_bytes(b"glb")
        return glb_path

    monkeypatch.setattr(truerender_app, "segment_and_crop", fake_segment_and_crop)
    monkeypatch.setattr(truerender_app, "generate_mesh", fake_generate_mesh)
    monkeypatch.setattr(truerender_app, "convert_obj_to_glb", fake_convert_obj_to_glb)

    with truerender_app.jobs_lock:
        truerender_app.jobs.clear()
        truerender_app.active_job_id = None

    image = BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")
    image.seek(0)

    client = TestClient(truerender_app.app)
    response = client.post(
        "/reconstruct/image",
        data={"prompt": "the main object"},
        files={"file": ("sample.png", image, "image/png")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job_response = client.get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    job = job_response.json()
    assert job["status"] == "done"
    assert job["obj_url"].endswith("mesh.obj")
    assert job["glb_url"].endswith("mesh.glb")
