# TrueRender Technical Walkthrough Script

Target length: 7-8 minutes. Read the "Say" sections out loud. Use the "Show" lines as your screen plan. Do not read code line by line; point at the referenced ranges and explain what they prove.

## 0:00-0:45 — Project Goal And Final System

Show: `README.md` lines 1-7, then `assets/evaluation/truerender_v4_pipeline_diagram_with_model_types.svg`.

Say:

TrueRender answers one applied machine learning question: can an ordinary object image or phone video be converted into a usable 3D mesh without manual 3D modeling or cleanup?

The final project is a Colab-hosted web demo. A user uploads an object image, enters a text prompt like "bottle" or "shoe", and the system segments the prompted object with SAM 3, prepares a clean canonical crop, sends that crop through TripoSR for single-image 3D reconstruction, and returns downloadable OBJ and GLB files in the browser. The README summary at `README.md` lines 3-7 is the high-level description of the final system.

The important thing is that this is not only a notebook experiment. The repository contains a working FastAPI backend, a browser frontend, a Colab setup notebook, the production reconstruction pipeline, evaluation images, and the research notebooks that explain how I got to the final design.

## 0:45-1:45 — Final Pipeline Architecture

Show: `static/index.html` lines 153-168, `src/app.py` lines 84-117, then `src/app.py` lines 42-69.

Say:

The final pipeline starts in the web UI. In `static/index.html` lines 153-168, the frontend has a SAM prompt input, an image upload dropzone, a `model-viewer` preview, and download links for OBJ and GLB.

When the user uploads an image, `src/app.py` lines 84-117 define the `/reconstruct/image` endpoint. That endpoint accepts the uploaded file and prompt, creates a UUID job, stores job state, saves the input image, and starts the reconstruction as a background task.

The actual job lifecycle is in `src/app.py` lines 42-69. The backend first sets the stage to segmenting, calls `segment_and_crop`, then calls `generate_mesh`, then converts the OBJ to GLB and exposes the final `mesh_url`, `obj_url`, and `glb_url`. The status endpoint is `src/app.py` lines 120-126, and the frontend polling logic is `static/index.html` lines 200-242.

So the app is deliberately split into three responsibilities: `static/index.html` handles user interaction and polling, `src/app.py` handles jobs and HTTP routes, and `src/pipeline.py` handles the ML pipeline.

## 1:45-2:55 — Production ML Pipeline

Show: `src/pipeline.py` lines 78-115, then lines 118-177, then lines 189-242.

Say:

The production ML logic lives in `src/pipeline.py`. The model loading path is `src/pipeline.py` lines 78-115. It lazily verifies the TripoSR runtime, patches TripoSR when needed, runs a child Python import probe, and then builds the SAM 3 video predictor. This matters because Colab package state can drift, so the app verifies the runtime before serving jobs.

The segmentation stage is `src/pipeline.py` lines 118-177. Even though the app uses a single uploaded image, it uses SAM 3's image/video session interface: it saves the image as `0.jpg`, starts a SAM session, adds the user's text prompt, propagates the mask, and saves both a binary mask and an RGBA image. The mask is cleaned with morphological open and close operations in lines 163-166 before it becomes the alpha channel.

The canonical crop logic is `src/pipeline.py` lines 189-242. It scores mask candidates by area, bounding box area, fill ratio, and center penalty, then chooses the best object crop. It pads the object, square-crops it, resizes it into a `768 x 768` transparent RGBA canvas, and saves `canonical_rgba.png`. That preprocessing is a key technical contribution because image-to-3D models are sensitive to background contamination, off-center objects, and clipped inputs.

## 2:55-3:45 — Mesh Generation And Browser Delivery

Show: `src/pipeline.py` lines 245-326 and `src/pipeline.py` lines 329-334.

Say:

Mesh generation is in `src/pipeline.py` lines 245-326. The canonical RGBA crop is composited onto a neutral gray RGB background in lines 273-280 because TripoSR expects an RGB input image. The code then runs TripoSR as a subprocess with `--no-remove-bg`, `--foreground-ratio 0.85`, OBJ export, and marching cubes resolution 256 in lines 283-292.

The `--no-remove-bg` flag is intentional. SAM 3 already produced the object mask, so I do not want TripoSR to run a second background-removal model and potentially create inconsistent masks. After TripoSR finishes, the app finds the OBJ output and returns it. Then `src/pipeline.py` lines 329-334 use `trimesh` to convert the OBJ into GLB for browser preview.

On the frontend side, `static/index.html` lines 212-242 poll `/jobs/{job_id}` every 1.5 seconds. Once the job is done, lines 226-232 set the `model-viewer` source and populate the OBJ and GLB download links.

## 3:45-4:55 — Where ML Is Applied

Show: `README.md` lines 35-41, then `ATTRIBUTION.md` lines 27-38.

Say:

The final demo uses two pretrained models for inference and integration: SAM 3 for text-guided segmentation, and TripoSR for single-image 3D mesh generation. I am careful to say "used pretrained models for inference and integration" here, not "trained SAM 3" or "trained TripoSR."

The research versions tested additional models and approaches. Grounding DINO was used to detect the object and produce a better phrase for SAM. VGGT-1B was used as a vision transformer for camera pose and point estimation. 3DGS and 2DGS were trained as reconstruction pipelines. SuGaR and TSDF were used as mesh extraction attempts from splat representations. Stable Fast 3D was attempted in v4 before the final TripoSR fallback.

The model attribution list is in `ATTRIBUTION.md` lines 27-38. The evaluation summary in `README.md` lines 35-41 shows the key comparison: v1 had strong render metrics but high runtime, v2 and v3 were faster but produced fragmented TSDF meshes, and v4 produced the usable final mesh path.

## 4:55-6:15 — Research Progression And Failure Analysis

Show: `assets/evaluation/truerender_research_progression_with_model_types.svg`, then `README.md` lines 35-43, then `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`.

Say:

The project changed direction based on evidence. Version 1 used COLMAP, SAM 3, 3D Gaussian Splatting, and SuGaR. It produced good reconstruction metrics: `notebooks/reconstruction_v1.ipynb` lines 6774-6778 show 3DGS training reaching L1 `0.002104` and PSNR `35.6169 dB`. The problem was practicality. The README summarizes v1 as about 130 minutes total at `README.md` lines 35-39, so it was too slow for an interactive demo.

Version 2 replaced slower pose recovery with VGGT, used Grounding DINO to find the object phrase, trained 2DGS, and tried TSDF extraction. The frame preprocessing is visible in `notebooks/reconstruction_v2.ipynb` lines 75-94: it extracts frames at 5 FPS and keeps the 30 sharpest frames. The 2DGS training metric is `notebooks/reconstruction_v2.ipynb` lines 1189-1193, with L1 `0.004936` and PSNR `30.5943 dB`.

But the mesh failed. `notebooks/reconstruction_v2.ipynb` lines 1428-1435 show TSDF extraction producing `45,944` clusters, and lines 1466-1471 explain the root cause: white-background pixels with undefined or garbage depth were being fused as real surfaces.

Version 3 tested the hypothesis that white JPG masks were the problem. `notebooks/reconstruction_v3.ipynb` lines 1-23 state the v3 goal: keep SAM masks as real alpha masks and train 2DGS on RGBA PNGs. The SAM output evidence is `notebooks/reconstruction_v3.ipynb` lines 896-900: 30 out of 30 frames were masked, 30 RGBA frames were saved, and the sample mode was RGBA.

Even after that, v3 failed the mesh quality gate. `notebooks/reconstruction_v3.ipynb` lines 2889-2898 show 50 connected components, largest component ratio `0.842`, and `MESH_OK: False`. That result is the major lesson of the project: a good render metric does not guarantee a usable mesh.

Version 4 made the clean-mesh-first pivot. `notebooks/reconstruction_v4.ipynb` lines 1-23 explain the decision to stop relying on splat-to-TSDF mesh extraction and instead use SAM masks to select one clean canonical object image for image-to-3D generation.

## 6:15-7:15 — Preprocessing And Data Quality

Show: `notebooks/reconstruction_v4.ipynb` lines 415-417, lines 458-466, lines 496-499, and lines 543-574. Then show `src/pipeline.py` lines 163-166 and 189-242.

Say:

A lot of the quality comes from preprocessing, not just from choosing powerful models. The source data has common phone-video problems: blurry frames, background contamination, imperfect masks, and object crops that can be too small, off-center, or clipped.

The notebook evidence for v4 is clear. `notebooks/reconstruction_v4.ipynb` lines 415-417 show SAM outputs for 60 frames, with 60 masks and 60 RGBA frames. Lines 458-466 show the mask cleanup and alpha-channel creation. Lines 496-499 show the selected canonical frame and its score, including area fraction, bounding box area fraction, fill ratio, and center penalty. Lines 543-574 show the scoring, crop, resize, and canonical RGBA export.

The production app mirrors this logic in `src/pipeline.py`: mask cleanup is lines 163-166, candidate scoring is lines 189-216, and the final `768 x 768` transparent crop is lines 218-242. That means the final app is not a loose notebook copy; it ports the important data-quality work into reusable production code.

## 7:15-8:10 — Engineering Challenges In Colab

Show: `setup_colab.ipynb` lines 1-19, lines 55-115, lines 175-255, and lines 257-320. Then show `ATTRIBUTION.md` lines 19-25.

Say:

The biggest engineering challenge was making research models work together in Colab. The setup notebook handles Hugging Face access, dependency pins, SAM 3 installation, TripoSR installation, compatibility patches, runtime probes, and launching a public demo URL.

In `setup_colab.ipynb` lines 55-115, the setup installs binary-sensitive packages like NumPy, OpenCV, Pillow, Transformers, and FastAPI, logs into Hugging Face, and installs SAM 3. Lines 175-255 patch TripoSR so `rembg` and `onnxruntime` are not imported unnecessarily when `--no-remove-bg` is used, then force-reinstall critical packages and run a child Python import probe.

The launch cell is `setup_colab.ipynb` lines 257-320. It starts Uvicorn in a clean Python subprocess and opens a Cloudflare Tunnel URL. So this is a Colab-hosted deployment, not a permanent production server.

The debugging history is summarized in `ATTRIBUTION.md` lines 19-25: Transformers drift broke Grounding DINO, Matplotlib API changes broke 2DGS visualization utilities, TSDF fused bad background depths, NumPy ABI issues required pinning, and TripoSR needed lazy `rembg` imports.

## 8:10-9:00 — Codebase Tour And Closing

Show: quick tabs in this order: `README.md`, `src/app.py`, `src/pipeline.py`, `static/index.html`, `setup_colab.ipynb`, `notebooks/reconstruction_v3.ipynb`, `notebooks/reconstruction_v4.ipynb`, `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`, and `examples/meshes/meshv4_finalmeshexample_hydroflask.obj`.

Say:

For a grader reading the repo, the split is intentional. `README.md` lines 1-7 give the project summary, lines 33-43 give the evaluation comparison, and lines 45-63 link the diagrams and mesh evidence. `src/app.py` is the FastAPI job system. `src/pipeline.py` is the production reconstruction pipeline. `static/index.html` is the browser UI. `setup_colab.ipynb` is the deployment and environment setup.

The notebooks are the research record. `notebooks/reconstruction_v1.ipynb` documents the COLMAP plus 3DGS plus SuGaR baseline. `notebooks/reconstruction_v2.ipynb` documents Grounding DINO, SAM 3, VGGT, 2DGS, and the TSDF failure. `notebooks/reconstruction_v3.ipynb` documents real alpha masks and the mesh quality gate. `notebooks/reconstruction_v4.ipynb` documents the clean-mesh-first pivot and the final TripoSR path.

To close: TrueRender combines segmentation, preprocessing, image-to-3D generation, evaluation, and deployment into one applied ML system. The final app uses SAM 3 to isolate a prompted object and TripoSR to generate a downloadable mesh. The main technical lesson was that reconstruction metrics alone were not enough. I had to evaluate actual mesh usability, diagnose fragmented geometry, and pivot from splat-to-mesh extraction to a cleaner image-to-3D pipeline.

The strongest contributions are the multi-stage ML pipeline, the model comparison across v1 through v4, the use of segmentation and vision-transformer-based geometry experiments, the preprocessing and failure analysis, and the functional Colab-hosted web deployment.

## Recording Checklist

- Open `README.md` at lines 1-7 and 33-63.
- Open `assets/evaluation/truerender_v4_pipeline_diagram_with_model_types.svg`.
- Open the running TrueRender web demo if available.
- Open `src/app.py` at lines 42-69 and 84-126.
- Open `src/pipeline.py` at lines 78-115, 118-242, and 245-334.
- Open `static/index.html` at lines 153-168 and 200-242.
- Open `notebooks/reconstruction_v3.ipynb` around lines 2889-2898 for the failed mesh gate.
- Open `notebooks/reconstruction_v4.ipynb` around lines 496-574 and 1189-1303 for canonical crop and final mesh evidence.
- Open `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`.
- End on the browser `model-viewer` preview or `examples/meshes/meshv4_finalmeshexample_hydroflask.obj`.

## Phrases To Use Carefully

- Say "used pretrained models for inference and integration" for SAM 3 and TripoSR.
- Say "trained 3DGS and 2DGS reconstruction pipelines" only for v1-v3.
- Say "v4 produces a cleaner usable mesh" rather than "v4 has better PSNR."
- Say "Colab-hosted deployment through Cloudflare Tunnel" rather than permanent production deployment.
- Say "AI-assisted development was reviewed, modified, and debugged" and cite `ATTRIBUTION.md` lines 3-25.
