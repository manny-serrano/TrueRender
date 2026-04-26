# TrueRender Localhost Demo

This MVP wraps the `reconstruction_v4.ipynb` SAM 3 + TripoSR path in a single FastAPI app. It accepts one image, segments the prompted object with SAM 3, applies the same canonical crop used in v4, runs patched TripoSR, converts the OBJ to GLB for browser preview, and exposes both download links.

## Colab Flow

1. Open `setup_colab.ipynb` in Google Colab with an A100 GPU runtime.
2. Run Cell 1 from the TrueRender repo root. It installs the web app dependencies, installs SAM 3, installs TripoSR, and applies the same lazy-`rembg` TripoSR patches used in `reconstruction_v4.ipynb`.
3. Run Cell 2. It starts `uvicorn` on port 8000 in a background thread, opens an ngrok tunnel, and prints the public URL.
4. Open the printed URL in a new tab, enter a SAM prompt such as `green water bottle`, and drop in a JPG or PNG.

The UI should progress through `segmenting`, `cropping`, `generating mesh`, and `done`. When complete, it displays the GLB in `<model-viewer>` and provides `Download OBJ` and `Download GLB` links.

## Files

- `app.py` contains the FastAPI app, in-memory job tracking, and output serving.
- `pipeline.py` contains lazy SAM 3 loading, the copied v4 canonical crop logic, the copied v4 TripoSR patches, TripoSR subprocess inference, and OBJ-to-GLB conversion.
- `static/index.html` is the no-build drag-drop frontend.
- `outputs/` stores per-job generated meshes and is gitignored.

## Notes

The demo is intentionally single-process and single-job-at-a-time. If a reconstruction is already running, a second upload returns HTTP 429. Models are not loaded at Python import time; the first reconstruction request initializes SAM 3 and verifies the patched TripoSR runtime.
