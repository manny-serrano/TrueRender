# TrueRender Technical Walkthrough

This document is a recording guide for the 5-10 minute technical walkthrough video. The intended audience is a fellow ML engineer or grader who wants to understand how the project works, where the machine learning components are applied, what failed during development, and where the significant technical contributions live in the repository.

Recommended recording format:

- Show the repository and selected files, but do not spend time reading code line by line.
- Use the README diagrams and evaluation images to orient the viewer.
- Open the key implementation files when explaining the final app.
- Open the notebooks only to show the research progression and evidence outputs.
- Keep the tone technical, but explain why each design choice mattered.

Target length: 7-8 minutes.

## 0:00-0:45 - Project Goal And Final System

Start by stating the central project question:

> Can an ordinary object image or phone video be converted into a usable 3D mesh through an applied ML pipeline, without manual 3D modeling or cleanup?

Explain that TrueRender is a Colab-hosted web app. The final implementation takes an uploaded object image, segments the prompted object, prepares a canonical transparent crop, runs image-to-3D mesh generation, and serves downloadable `OBJ` and `GLB` files through a browser UI.

Show:

- `README.md`
- `assets/evaluation/truerender_v4_pipeline_diagram_with_model_types.svg`
- `static/index.html` if showing the UI source briefly

Key point to say:

> The final product is not just a notebook experiment. It is a working ML application with a frontend, FastAPI backend, Colab setup, segmentation, mesh generation, and downloadable outputs.

## 0:45-1:45 - Final Pipeline Architecture

Walk through the final app pipeline from input to output.

Important files:

- `src/app.py`
- `src/pipeline.py`
- `setup_colab.ipynb`
- `static/index.html`

Pipeline stages:

1. The user enters a text prompt and uploads an image in the browser UI.
2. `src/app.py` accepts the request through `/reconstruct/image`.
3. The app creates a job ID, saves the uploaded image, and runs reconstruction in the background.
4. `segment_and_crop()` in `src/pipeline.py` loads SAM 3, starts a SAM video/image session, applies the user prompt, and saves a foreground alpha mask.
5. The masked object is scored and converted into a centered `768 x 768` canonical RGBA crop.
6. `generate_mesh()` composites that crop onto a neutral gray background and runs TripoSR with `--no-remove-bg`.
7. The resulting OBJ is converted to GLB for browser preview.
8. The frontend polls `/jobs/{job_id}` until the mesh URLs are ready.

Code references to mention:

- `src/app.py`: FastAPI routes, job state, background task flow.
- `src/pipeline.py`: `load_models()`, `segment_and_crop()`, `generate_mesh()`, `convert_obj_to_glb()`.
- `static/index.html`: prompt input, upload UI, polling, and `model-viewer` preview.

Key point to say:

> The ML pipeline is not hidden behind one black-box call. The app explicitly coordinates segmentation, preprocessing, mesh generation, validation-friendly packaging, and browser delivery.

## 1:45-2:45 - Where Machine Learning Is Applied

Explain the main ML models and what each contributes.

Final demo models:

- SAM 3: text-guided object segmentation.
- TripoSR: single-image-to-3D mesh generation.

Research-stage models:

- Grounding DINO: object detection in v2/v3 to produce a better object phrase for SAM.
- VGGT-1B: vision transformer for camera pose and point estimation.
- 3DGS and 2DGS: Gaussian splatting reconstruction.
- SuGaR and TSDF extraction: mesh extraction attempts from splat representations.
- Stable Fast 3D: attempted image-to-3D route in v4 before falling back to TripoSR.

Important evidence locations:

- `notebooks/reconstruction_v2.ipynb`: Grounding DINO, SAM 3, VGGT, 2DGS.
- `notebooks/reconstruction_v3.ipynb`: alpha-mask pipeline and TSDF quality gate.
- `notebooks/reconstruction_v4.ipynb`: clean-mesh-first pivot, canonical frame selection, SF3D attempt, TripoSR fallback.
- `src/pipeline.py`: production SAM 3 plus TripoSR path.

Key point to say:

> The final app is simple for the user, but it came from testing a sequence of ML approaches: multi-view reconstruction, transformer-based pose estimation, splat-based training, segmentation-aware masking, and finally image-to-3D mesh generation.

## 2:45-4:00 - Research Progression And Iterations

Use this section to show that the project was driven by evaluation results, not just random tool chaining.

Show:

- `README.md` Evaluation section.
- `assets/evaluation/truerender_research_progression_with_model_types.svg`
- `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`
- `notebooks/reconstruction_v3.ipynb` cell 0.
- `notebooks/reconstruction_v4.ipynb` cell 0.

Explain each iteration:

### v1: COLMAP + SAM 3 + 3DGS + SuGaR

What worked:

- Produced recognizable object geometry.
- Training metrics were strong: about `35.6 dB PSNR` and `L1 = 0.00210`.

What failed:

- Full pipeline was too slow for an interactive demo.
- It was not practical as the final product path.

### v2: Grounding DINO + SAM 3 + VGGT + 2DGS + TSDF

What changed:

- Replaced slower COLMAP pose recovery with VGGT.
- Used Grounding DINO to detect the target and improve the SAM prompt.
- Used 2DGS and TSDF extraction for a faster mesh route.

What failed:

- Faster runtime, but mesh extraction shattered.
- The notebook documents tens of thousands of disconnected TSDF clusters in the failure case.

### v3: Real Alpha Masks For 2DGS

What changed:

- The main hypothesis was that v2 failed because white-background JPG masks polluted TSDF fusion.
- v3 preserved SAM masks as real RGBA alpha channels.
- 2DGS was trained on alpha-aware inputs.

What was measured:

- The TSDF quality gate still reported disconnected geometry.
- v3 failed the mesh quality gate, so it was not accepted as the final route.

### v4: Clean Mesh First

What changed:

- Pivoted away from splat-to-TSDF mesh extraction.
- Used SAM 3 to select and crop a clean canonical object image.
- Used image-to-3D generation to produce a coherent final mesh.

What improved:

- Final mesh was compact, usable, and browser-previewable.
- `notebooks/reconstruction_v4.ipynb` records a `3.9 MB` final OBJ passing the validation/package step.

Key point to say:

> The most important technical decision was realizing that high image reconstruction quality did not guarantee a clean mesh. For this project objective, mesh usability mattered more than PSNR alone.

## 4:00-5:10 - Preprocessing And Data Quality

Explain why preprocessing was a major part of the project.

Data quality problems:

- Blurry video frames.
- Background contamination.
- Object masks with holes or noisy edges.
- Crops where the object is too small, off-center, or clipped.
- Colab dependency drift causing binary package failures.

Preprocessing solutions:

- Laplacian sharpness scoring to keep the clearest frames in v2/v3/v4.
- SAM 3 masks converted to alpha channels rather than white-background images.
- Morphological open/close operations on masks.
- Canonical crop scoring based on mask area, bounding box area, fill ratio, and center penalty.
- Final `768 x 768` transparent RGBA object crop.
- Neutral gray compositing before TripoSR because TripoSR expects an RGB object image.

Code references:

- `notebooks/reconstruction_v2.ipynb` cell 5: frame extraction and sharpness filtering.
- `notebooks/reconstruction_v3.ipynb` cell 13: real masks and RGBA training images.
- `notebooks/reconstruction_v4.ipynb` cell 11: canonical crop scoring.
- `src/pipeline.py`: production version of SAM mask cleanup and canonical crop creation.

Key point to say:

> A lot of the project quality came from data preparation. The ML models are powerful, but the output depends heavily on giving them the right object crop and suppressing background artifacts.

## 5:10-6:10 - Evaluation And Failure Analysis

Show how evaluation was done.

Quantitative evidence:

- v1: `PSNR 35.6 dB`, `L1 = 0.00210`.
- v2/v3: `PSNR 30.6 dB`, `L1 = 0.00494`.
- v3 mesh gate: `50` connected components, largest component ratio around `0.842`, final `MESH_OK: False`.
- v4 final mesh: `mesh.obj OK 3.9 MB`.

Qualitative evidence:

- Side-by-side mesh visualizations in `notebooks/reconstruction_v3.ipynb`.
- Final comparison image at `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`.
- Browser preview through `model-viewer`.

What to emphasize:

- PSNR and L1 measure image reconstruction quality, but the product objective is a usable 3D mesh.
- Mesh connected components and visual inspection were necessary because a mesh can have a file output and still be unusable.
- The project used evaluation results to justify the v4 pivot.

Key point to say:

> The evaluation changed the project direction. The model with the best render metric was not automatically the best user-facing solution.

## 6:10-7:15 - Engineering Challenges

Discuss the most important technical challenges.

### Dependency Drift In Colab

Problems encountered:

- NumPy ABI mismatch with compiled packages.
- Transformers version drift breaking Grounding DINO.
- Matplotlib API changes breaking 2DGS.
- `rembg` and `onnxruntime` imports causing unnecessary failures in TripoSR.

Fixes:

- `setup_colab.ipynb` pins binary-sensitive dependencies.
- SAM 3 install is patched for Python 3.12 compatibility.
- TripoSR is patched so `rembg` is imported lazily only when background removal is actually used.
- The setup cell runs child Python probes to verify imports.
- The launch cell runs Uvicorn in a clean Python subprocess so the app does not inherit stale notebook imports.

Relevant files:

- `setup_colab.ipynb`
- `src/pipeline.py`
- `ATTRIBUTION.md`

### Notebook To App Transition

Challenges:

- Research notebooks were exploratory and long-running.
- The final app needed a reusable, narrow path.
- The pipeline had to expose progress and downloadable outputs.
- The mesh path had to be robust enough for a short demo.

Fix:

- Core logic was consolidated into `src/pipeline.py`.
- Web job orchestration lives in `src/app.py`.
- The UI lives in `static/index.html`.

Key point to say:

> A major technical contribution was not only running models, but making them work together reliably in a constrained Colab deployment environment.

## 7:15-8:15 - Codebase Tour For Grader

Use this section as a quick map of where to look.

Important files:

- `README.md`: project summary, quick start, evaluation table, diagrams, and final mesh evidence.
- `SETUP.md`: setup instructions for running the Colab demo.
- `ATTRIBUTION.md`: AI tooling and third-party model attribution.
- `setup_colab.ipynb`: install, patch, verify, and launch the app.
- `src/app.py`: FastAPI routes and job lifecycle.
- `src/pipeline.py`: production reconstruction pipeline.
- `static/index.html`: user interface and polling logic.
- `notebooks/reconstruction_v1.ipynb`: COLMAP + 3DGS + SuGaR baseline.
- `notebooks/reconstruction_v2.ipynb`: VGGT + 2DGS + TSDF speed-focused attempt.
- `notebooks/reconstruction_v3.ipynb`: alpha-mask TSDF attempt and quality gate.
- `notebooks/reconstruction_v4.ipynb`: clean-mesh-first path and final TripoSR result.
- `assets/evaluation/`: visual comparisons and pipeline diagrams.
- `examples/meshes/`: example final mesh output.

Key point to say:

> The repo separates the final demo path from the research notebooks. The notebooks show the experimentation and evaluation history, while `src/` and `static/` contain the runnable application.

## 8:15-9:00 - Closing Summary

Close with a concise technical summary:

> TrueRender combines segmentation, preprocessing, image-to-3D generation, evaluation, and deployment into one applied ML system. The final app uses SAM 3 to isolate a prompted object and TripoSR to generate a downloadable mesh. The main lesson from the project was that reconstruction metrics alone were not enough. I had to evaluate the actual mesh quality, diagnose failure cases, and pivot from splat-to-mesh extraction to a cleaner image-to-3D pipeline.

Mention the strongest rubric-relevant contributions:

- Multi-stage ML pipeline.
- Vision transformer and object detection experiments.
- Image segmentation model on custom data.
- Quantitative and qualitative evaluation.
- Error analysis and iteration across v1-v4.
- Functional web deployment.
- Substantial debugging and integration work.

## Optional Screen Recording Checklist

Before recording, open these tabs:

- `README.md`
- `src/app.py`
- `src/pipeline.py`
- `setup_colab.ipynb`
- `notebooks/reconstruction_v3.ipynb`
- `notebooks/reconstruction_v4.ipynb`
- `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`
- The running TrueRender web demo, if available

Recommended visual sequence:

1. README title and What it Does.
2. Pipeline diagram.
3. Running app demo.
4. `src/app.py` route overview.
5. `src/pipeline.py` functions.
6. v3 failure evidence.
7. v4 final output evidence.
8. Evaluation image.
9. Closing on final mesh preview.

## Claims To Make Carefully

Use precise wording in the video:

- Say "used pretrained models for inference and integration" rather than "trained SAM 3" or "trained TripoSR."
- Say "trained 3DGS/2DGS reconstruction pipelines" only when discussing v1-v3.
- Say "v4 produces a cleaner usable mesh" rather than "v4 is quantitatively better on PSNR", because v4 is evaluated mainly by mesh validity and usability.
- Say "Colab-hosted deployment" rather than permanent production deployment.
- Say "AI-assisted development was reviewed, modified, and debugged" and point to `ATTRIBUTION.md`.

