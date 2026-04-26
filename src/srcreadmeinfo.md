# Source Code

The runnable FastAPI demo source lives here:

- `app.py`: API routes, job tracking, static file serving, and output downloads.
- `pipeline.py`: SAM 3 segmentation, canonical crop generation, TripoSR mesh generation, and OBJ-to-GLB conversion.

`setup_colab.ipynb` launches the app with `uvicorn src.app:app`.
