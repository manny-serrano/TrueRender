# TrueRender Pipeline System Overview

This file summarizes the architecture across the four reconstruction notebooks. Each notebook represents a different version of the project pipeline, with different reconstruction strategies, model choices, preprocessing, mesh extraction methods, and failure modes.

Cell references below use the notebook's Jupyter cell order and are approximate where the notebook contains repeated setup or troubleshooting cells.

## Research Goal

TrueRender asks whether an ordinary object capture can be converted into a usable 3D asset with minimal manual cleanup. The project starts with multi-view reconstruction from a phone video, then iterates toward a cleaner final demo that uses segmentation plus single-image mesh generation.

---

## Version 1: COLMAP + SAM 3 + 3DGS + SuGaR

Notebook: `notebooks/reconstruction_v1.ipynb`

### Pipeline Goal

The first notebook tests a full multi-view reconstruction pipeline: phone video to extracted frames, COLMAP camera reconstruction, 3D Gaussian Splatting, object masking with SAM 3, and mesh extraction with SuGaR.

### Input Data

- Source video: `hydroflaskgreen.MOV` from Google Drive.
- Frames extracted at approximately 5 FPS.
- Target object: green Hydro Flask.

### Architecture

1. Extract frames from the phone video with `ffmpeg`.
2. Run COLMAP feature extraction, matching, sparse reconstruction, and image undistortion.
3. Train 3D Gaussian Splatting on the reconstructed scene.
4. Use SAM 3 video segmentation with the prompt `green hydroflask water bottle`.
5. Apply the propagated SAM masks to isolate the object by replacing background pixels with white.
6. Retrain or run 3DGS on the masked object-focused image set.
7. Use SuGaR to extract a mesh from the 3DGS representation.
8. Export and save the refined mesh outputs to Google Drive.

### Models and Tools

- COLMAP: camera pose estimation and sparse structure-from-motion.
- SAM 3: prompt-based object segmentation and video mask propagation.
- 3D Gaussian Splatting (`graphdeco-inria/gaussian-splatting`): neural scene representation.
- SuGaR (`Anttwo/SuGaR`): surface-aligned mesh extraction from 3DGS.
- `trimesh` and Open3D-style tooling for mesh inspection/export.

### Data Cleaning and Preprocessing

- Basic frame extraction only.
- No sharpness filtering in v1.
- Object masks are baked into white-background RGB images instead of real alpha masks.
- COLMAP outputs are checkpointed to Drive so long reconstruction steps do not need to be rerun.

### Outputs

- COLMAP reconstruction folders.
- 3DGS checkpoint and point cloud.
- Rendered 3DGS views.
- SuGaR refined mesh assets.

### Notes and Limitations

v1 produced usable geometry, but it was slow because COLMAP, 3DGS, and SuGaR together took roughly two hours. The white-background masking strategy also became a source of problems in later TSDF-based experiments.

---

## Version 2: Grounding DINO + SAM 3 + VGGT + 2DGS + TSDF

Notebook: `notebooks/reconstruction_v2.ipynb`

### Pipeline Goal

v2 upgrades the pipeline for speed and automation. It replaces COLMAP with VGGT camera estimation, adds Grounding DINO for automatic object detection, uses SAM 3 for segmentation, trains 2D Gaussian Splatting, and attempts mesh extraction through TSDF fusion.

### Input Data

- Source video: `hydroflaskgreen.MOV`.
- Frames extracted at 5 FPS.
- The 30 sharpest frames are kept for the reconstruction pipeline.

### Architecture

1. Extract frames from the video.
2. Score frames by variance of Laplacian and keep the sharpest 30.
3. Run Grounding DINO on the first frame to detect the target object.
4. Convert the DINO label into a richer SAM 3 text prompt, such as `green bottle container water bottle`.
5. Run SAM 3 video segmentation and save masked frames.
6. Run VGGT to estimate camera poses and 3D points.
7. Export VGGT predictions into a COLMAP-style dataset for downstream training.
8. Scale VGGT intrinsics from 518 x 518 model resolution back to the original 1080 x 1920 image resolution.
9. Swap the original COLMAP image folder with the SAM-masked frames.
10. Train 2D Gaussian Splatting for 7,000 iterations.
11. Run 2DGS TSDF extraction with `render.py --depth_ratio 1`.
12. Patch 2DGS mesh extraction when white-background pixels create invalid depth surfaces.
13. Add a Screened Poisson fallback from opacity-filtered Gaussian centers if TSDF remains fragmented.

### Models and Tools

- Grounding DINO (`IDEA-Research/GroundingDINO`): text-prompted object detection.
- SAM 3 (`facebookresearch/sam3`): video segmentation.
- VGGT (`facebookresearch/vggt`, `facebook/VGGT-1B`): feed-forward camera pose and 3D point estimation.
- 2D Gaussian Splatting (`hbb1/2d-gaussian-splatting`): surface-aligned splat training.
- Open3D: TSDF/Poisson mesh processing and connected-component analysis.

### Data Cleaning and Preprocessing

- Laplacian sharpness filtering removes blurry frames.
- DINO automatically chooses the target object instead of relying only on a fixed prompt.
- SAM masks are saved as white-background images.
- VGGT intrinsics are corrected from model input size to original image size.

### Outputs

- 30 selected sharp frames.
- Grounding DINO detection result and SAM 3 masks.
- VGGT COLMAP-style sparse reconstruction.
- 2DGS trained output.
- TSDF mesh attempt.
- Poisson fallback mesh if TSDF is unusable.

### Debugging and Failure Notes

The main v2 failure was mesh fragmentation. TSDF fusion interpreted white background pixels as valid geometry because those pixels still contributed depth. The notebook patches `mesh_utils.py` to mask near-white pixels, but the TSDF output could still produce many disconnected components. This led to the Poisson fallback in the next cell.

v2 also needed dependency patches, including a Matplotlib `tostring_rgb()` to `buffer_rgba()` compatibility fix for newer Matplotlib versions.

---

## Version 3: Alpha-Mask VGGT + 2DGS Pipeline

Notebook: `notebooks/reconstruction_v3.ipynb`

### Pipeline Goal

v3 keeps the faster VGGT + 2DGS architecture from v2 but reworks masking. Instead of baking the SAM masks into white-background JPGs, it saves real binary masks and RGBA PNG training images so 2DGS can use alpha masks during TSDF fusion.

### Input Data

- Source video: `hydroflaskgreen.MOV`.
- 30 sharp frames selected by Laplacian filtering.
- Original RGB frames are preserved for VGGT camera estimation.
- RGBA masked frames are used for 2DGS training.

### Architecture

1. Configure paths, Google Drive checkpoints, and Hugging Face credentials.
2. Extract and sharpness-filter frames.
3. Use Grounding DINO to detect the target object on the first filtered frame.
4. Pin and patch Transformers/Grounding DINO compatibility.
5. Use SAM 3 to propagate object masks across frames.
6. Save both binary mask PNGs and RGBA training PNGs.
7. Run VGGT on original RGB frames for pose estimation.
8. Export a COLMAP-style dataset where `images.txt` points to the RGBA PNG filenames.
9. Train 2DGS on the RGBA dataset.
10. Run TSDF mesh extraction with the expectation that `gt_alpha_mask` suppresses background depth.
11. Run a mesh quality gate using Open3D connected-component statistics and visual approval.
12. If TSDF fails, fall back to the earlier v1 SuGaR mesh.
13. Export final OBJ/STL/PLY assets and a source manifest.

### Models and Tools

- Grounding DINO: first-frame object detection.
- SAM 3: object mask propagation.
- VGGT / `facebook/VGGT-1B`: camera pose and point estimation.
- 2DGS: surface-aligned splat training and TSDF extraction.
- Open3D: mesh quality checks and export.
- Plotly/Matplotlib: side-by-side mesh comparison visualizations.

### Data Cleaning and Preprocessing

- Laplacian frame filtering.
- Prompt enrichment from DINO output to a more specific SAM 3 prompt.
- Real binary masks saved separately from RGB images.
- RGBA training images preserve foreground alpha.
- 2DGS training asserts that images remain RGBA before training starts.

### Outputs

- `frames_v3`: selected RGB frames.
- `masks_v3`: binary SAM masks.
- `frames_rgba_v3`: alpha-preserving training frames.
- `colmap_vggt_v3`: VGGT camera export in COLMAP-style format.
- `output_2dgs_v3`: trained 2DGS output.
- `outputs_v3`: final exported assets and manifest.

### Debugging and Failure Notes

v3 includes the major Grounding DINO / Transformers compatibility fix. Colab environments with newer Transformers broke Grounding DINO because `BertModel.get_head_mask` was missing or changed. The notebook force-pins `transformers==4.41.2`, verifies the imported version in a child process, and patches Grounding DINO's `bertwarper.py` if necessary.

The 2DGS Matplotlib patch from v2 is retained and made safer. The notebook checks for the old `tostring_rgb()` call, an earlier bad patch shape, or an already-patched `buffer_rgba()` version.

Even with real alpha masks, the v3 TSDF mesh still failed the quality gate. The notebook therefore keeps an explicit fallback path to the v1 SuGaR mesh and documents that textured OBJ assets should not be round-tripped through Open3D if material preservation matters.

---

## Version 4: Clean Mesh First, SAM 3 + Image-to-3D

Notebook: `notebooks/reconstruction_v4.ipynb`

### Pipeline Goal

v4 changes the project direction. Instead of trying to mesh splats from multi-view reconstruction, it uses the video only to select a clean object image, then runs an image-to-3D model to generate a coherent mesh. This is the final demo architecture.

### Input Data

- Source video: `hydroflaskgreen.MOV`.
- 60 candidate frames selected from the video.
- A single canonical SAM-masked image is selected as the final mesh input.

### Architecture

1. Establish a stable Colab dependency baseline before importing OpenCV, SAM, or image-to-3D packages.
2. Pin `numpy==1.26.4` and `opencv-python-headless==4.10.0.84`.
3. Restart the kernel if NumPy has already been imported with an incompatible ABI.
4. Extract more candidate frames than prior versions.
5. Score frames by Laplacian sharpness and keep 60 candidates.
6. Run SAM 3 with a fixed Hydro Flask prompt.
7. Save binary masks and RGBA frames.
8. Apply light morphological mask cleanup.
9. Score masked frames for canonical object quality using area, bounding-box fill, and center penalty.
10. Crop and resize the best object frame into a canonical RGBA image.
11. Attempt Stable Fast 3D as the preferred image-to-3D path.
12. If Stable Fast 3D is unavailable or fails, run TripoSR.
13. Composite the RGBA object onto a neutral gray background for TripoSR.
14. Run TripoSR with `--no-remove-bg`, `--foreground-ratio 0.85`, OBJ export, and marching cubes resolution 256.
15. Validate and package the final mesh.

### Models and Tools

- SAM 3: object segmentation.
- Stable Fast 3D (`Stability-AI/stable-fast-3d`): attempted preferred single-image mesh generator.
- TripoSR (`VAST-AI-Research/TripoSR`): final fallback and practical mesh generator.
- Open3D / mesh utilities: validation and packaging.
- FastAPI demo app in the repo uses the same final SAM 3 + TripoSR concept.

### Data Cleaning and Preprocessing

- More candidate frames are kept so the canonical image selector has better options.
- Laplacian filtering still removes blurry frames.
- SAM masks are preserved as alpha.
- Morphological open/close cleanup improves mask quality.
- The selected object is square-cropped, padded, resized, and composited over neutral gray for TripoSR.
- Background removal is disabled because SAM already provides the object mask.

### Outputs

- `frames_v4`: candidate frames.
- `masks_v4`: segmentation masks.
- `frames_rgba_v4`: alpha-preserving object frames.
- `canonical_v4`: selected single-image mesh input.
- `outputs_v4`: final mesh assets.
- `source_manifest.json`: metadata about the selected output and validation.

### Debugging and Failure Notes

v4 introduces a dependency baseline cell because Colab package updates caused NumPy ABI failures such as `numpy.dtype size changed`. The notebook pins NumPy and OpenCV before importing binary packages and restarts the runtime when needed.

Stable Fast 3D was attempted, but the notebook records that it can fail on Colab/Python 3.12 due to compiled dependency wheels. The project therefore uses TripoSR as the reliable final path.

TripoSR required patching because it imports `rembg` and ONNX runtime even when `--no-remove-bg` is used. The notebook patches both `run.py` and `tsr/utils.py` so `rembg` is imported lazily only if background removal is actually requested.

---

## Cross-Version Comparison

| Version | Camera / Pose Strategy | Segmentation Strategy | Reconstruction Strategy | Mesh Strategy | Main Lesson |
|---|---|---|---|---|---|
| v1 | COLMAP SfM | SAM 3 fixed prompt, white-background masked frames | 3DGS | SuGaR | Usable but very slow. |
| v2 | VGGT exported to COLMAP-style format | Grounding DINO + SAM 3, white-background masks | 2DGS | TSDF plus Poisson fallback | Faster, but white backgrounds broke TSDF depth fusion. |
| v3 | VGGT exported to COLMAP-style format | Grounding DINO + SAM 3, real RGBA masks | 2DGS | TSDF with SuGaR fallback | Alpha masks improved the pipeline but TSDF still failed visual quality. |
| v4 | None; single-image mesh generation | SAM 3 on selected canonical frame | TripoSR / attempted SF3D | Direct image-to-3D mesh | Best practical demo path: cleaner mesh and much faster interaction. |

## Final System Direction

The final submitted demo follows the v4 architecture:

1. User uploads a single object image and prompt.
2. SAM 3 segments the prompted object.
3. The segmented object is cropped and prepared as a clean image-to-3D input.
4. TripoSR generates the mesh.
5. The FastAPI app serves a browser preview and downloadable OBJ/GLB outputs.

The earlier notebooks are still important because they document the research path: the project tested true multi-view reconstruction first, discovered the mesh extraction failure modes, and then pivoted to a cleaner single-image mesh-generation system for the final product.
