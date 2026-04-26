# TrueRender — Self-Assessment Evidence

This document provides specific code/file evidence for each Category 1 claim in the CS 372 self-assessment.

Note: the prompt refers to `reconstruction.ipynb`; in this repository the first notebook is named `notebooks/reconstruction_v1.ipynb`.

## Claims

### 1. Modular code design with reusable functions and classes (3 pts)

**Evidence:**
- `notebooks/reconstruction_v3.ipynb` cell 2: reusable notebook helpers `reset_dir`, `copytree_fresh`, `restore_checkpoint`, and `assert_nonempty_dir` are defined for checkpointing and validation.
- `notebooks/reconstruction_v4.ipynb` cell 4: the same helper pattern is reused for the clean-mesh-first pipeline.
- `notebooks/reconstruction_v2.ipynb` cell 17: `run_sam3(prompt)` encapsulates SAM 3 prompt execution and is reused inside a retry loop.
- `src/pipeline.py` lines 24-28: reusable `reset_dir()`.
- `src/pipeline.py` lines 78-108: reusable `load_models()` for lazy SAM 3 / TripoSR setup.
- `src/pipeline.py` lines 111-235: `segment_and_crop()` encapsulates SAM 3 segmentation, RGBA mask saving, canonical crop scoring, and 768 x 768 export.
- `src/pipeline.py` lines 238-319: `generate_mesh()` encapsulates TripoSR execution.
- `src/pipeline.py` lines 322-327: `convert_obj_to_glb()` isolates preview conversion.

**Status:** STRONG

---

### 2. Data augmentation with evaluation of impact (5 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` cell 5 / lines 91-95: frame selection keeps `30 / 70` extracted frames using variance-of-Laplacian sharpness; recorded sharpness is `best: 199.0`, `cutoff: 48.9`, `worst dropped: 14.4`.
- `notebooks/reconstruction_v2.ipynb` cell 5 / lines 118-132: code computes `cv2.Laplacian(...).var()` and sorts frames by sharpness.
- `notebooks/reconstruction_v4.ipynb` cell 6 / lines 218, 231-232, 252-259: v4 keeps `60` candidate frames using the same Laplacian idea because the goal changed to selecting the best single image for image-to-3D generation.
- Impact comparison: v2/v3 were faster than v1 but TSDF quality failed; v4 uses more candidates and canonical selection, then produces a clean final mesh path. README lines 37-41 summarize the measured pipeline comparison.

**Notes:** This is stronger evidence for modality-specific data cleaning / frame selection than for classic data augmentation. There is no evidence of synthetic image augmentation such as flips, crops for training, color jitter, or multi-view augmentation with ablation. If the rubric strictly means augmentation, this claim should be swapped or reframed.

**Status:** WEAK — see notes

---

### 3. Basic preprocessing appropriate to the modality (3 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` cell 5 / lines 100-132: `ffmpeg` frame extraction at 5 FPS, Laplacian scoring, and frame selection.
- `notebooks/reconstruction_v1.ipynb` / lines 6207-6217: COLMAP `image_undistorter` converts images to COLMAP/PINHOLE-style undistorted layout.
- `notebooks/reconstruction_v2.ipynb` cell 22 / lines 816-852: VGGT intrinsics are scaled from model resolution `518 x 518` to original `1080 x 1920`; output says `cameras.txt rewritten with scaled intrinsics`.
- `notebooks/reconstruction_v3.ipynb` cell 17 / lines 1025-1027 and 1152-1157: VGGT outputs are exported in COLMAP format, with original-resolution scaling from 518.
- `notebooks/reconstruction_v4.ipynb` cell 11 / lines 515-573: selected object is square-cropped, padded, resized to `768`, and saved as canonical RGBA image.

**Status:** STRONG

---

### 4. Preprocessing pipeline addressing at least two substantive data quality challenges with documented impact (7 pts)

**Evidence:**
- Blurry frames: `notebooks/reconstruction_v2.ipynb` cell 5 / lines 91-95 keeps `30 / 70` frames and drops low-sharpness frames (`199.0` best vs `14.4` worst dropped).
- Background contamination: `notebooks/reconstruction_v3.ipynb` cell 0 / lines 10-23 explicitly identifies v2's white-background masks as the main failure and changes to real alpha masks.
- Real alpha masks: `notebooks/reconstruction_v3.ipynb` cell 13 / lines 896-900 records `masked 30/30 frames`, `Masks: 30  RGBA frames: 30`, and `Sample RGBA mode: RGBA`.
- Mask noise: `notebooks/reconstruction_v3.ipynb` cell 13 / lines 958-965 uses `cv2.MORPH_OPEN` and `cv2.MORPH_CLOSE` before saving masks and RGBA frames.
- Canonical frame quality: `notebooks/reconstruction_v4.ipynb` cell 11 / lines 496-497 records selected frame `frame_00005` with score `1.522`, `area_frac=0.1136`, `bbox_area_frac=0.1324`, `fill_ratio=0.8582`, and `center_penalty=0.0185`.
- Documented impact: `notebooks/reconstruction_v2.ipynb` lines 1431-1435 shows TSDF clustering failure with `#clusters=45944`; `notebooks/reconstruction_v3.ipynb` cell 25 records `Connected components: 50`, largest component ratio `0.842`, and `MESH_OK: False`; `notebooks/reconstruction_v4.ipynb` lines 1290-1303 records `mesh.obj OK 3.9 MB` and final assets saved.

**Status:** STRONG

---

### 5. Used a vision transformer for inference or as a frozen backbone (5 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` cell 25 / lines 907-910: output shows `Loading VGGT-1B...`, `Input: torch.Size([30, 3, 518, 518])`, `Cameras: 30`, and `3D points: 20000`.
- `notebooks/reconstruction_v2.ipynb` cell 25 / lines 940-948: loads `VGGT.from_pretrained("facebook/VGGT-1B")`, runs inference under `torch.no_grad()` and `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- `notebooks/reconstruction_v3.ipynb` cell 17 / lines 1144-1152: loads `facebook/VGGT-1B`, preprocesses images, and runs bfloat16 autocast inference.
- `notebooks/reconstruction_v2.ipynb` cell 23 / lines 861-864: notebook describes VGGT as replacing the COLMAP SfM pipeline with a single VGGT transformer forward pass.

**Status:** STRONG

---

### 6. Used an image segmentation model for inference on your data (5 pts)

**Evidence:**
- `notebooks/reconstruction_v1.ipynb` cell 24 / lines 6598-6607: builds SAM 3 video predictor, starts a session, adds prompt `green hydroflask water bottle`, and propagates masks across frames.
- `notebooks/reconstruction_v2.ipynb` cell 17 / lines 643-648: output shows `30/30 frames masked`, prompt `green bottle container water bottle`, and `Saved 30 masked frames (0 without mask)`.
- `notebooks/reconstruction_v3.ipynb` cell 13 / lines 896-900: output shows `masked 30/30 frames`, saved masks and RGBA frames, and `Sample RGBA mode: RGBA`.
- `notebooks/reconstruction_v4.ipynb` cell 9 / lines 411-415: output shows `Saved SAM outputs for 60 frames`, `Masks: 60  RGBA frames: 60`.
- `src/pipeline.py` lines 132-142: production demo uses `SAM_PREDICTOR.handle_request(...)`, `add_prompt`, and `propagate_in_video(...)` for inference.

**Double-counting check:** This claim is only image segmentation. It does not claim SAM 3 as a separate VLM claim.

**Status:** STRONG

---

### 7. Applied prompt engineering with evaluation of at least three prompt designs (3 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` cell 17 / lines 665-671 defines three prompt candidates:
  - `f"green {target_object} water bottle"`
  - `f"{target_object} on a wooden stool"`
  - `"green hydroflask water bottle"`
- `notebooks/reconstruction_v2.ipynb` cell 17 / lines 690-695 loops over the prompt candidates, calls `run_sam3(prompt)`, prints masked-frame counts, and stops when a prompt succeeds.
- `notebooks/reconstruction_v2.ipynb` cell 17 / lines 643-648 records the successful first prompt: `green bottle container water bottle`, `30/30 frames masked`.
- `notebooks/reconstruction_v3.ipynb` cell 13 / lines 918-922 defines the same three-candidate pattern, and lines 940-942 call `run_sam3(prompt)` for each candidate until success.
- `notebooks/reconstruction_v3.ipynb` cell 13 / lines 896-900 records successful prompt `green bottle container water bottle`, `masked 30/30 frames`.

**Notes:** The notebooks define and support three prompt designs, but because the first prompt succeeded, they do not show a full logged comparison across all three prompts. This is prompt-retry engineering with success criteria, not a complete prompt ablation table.

**Status:** WEAK — see notes

---

### 8. In ATTRIBUTION.md, provided a substantive account of how AI development tools were used (3 pts)

**Evidence:**
- `ATTRIBUTION.md` lines 3-5: identifies Cursor and ChatGPT/Claude-style prompting and explains how they were used.
- `ATTRIBUTION.md` lines 7-11: lists specific AI-assisted notebook sections and cells.
- `ATTRIBUTION.md` lines 13-17: describes code modified after AI generation, including v2 white-background masks to v3 alpha masks and VGGT intrinsic scaling.
- `ATTRIBUTION.md` lines 19-25: documents concrete debugging and rework: Transformers / `BertModel.get_head_mask`, Matplotlib `tostring_rgb()` to `buffer_rgba()`, TSDF white-pixel masking, Poisson fallback, NumPy ABI pin, and TripoSR `rembg` lazy imports.
- `src/app.py` lines 3-9 and `src/pipeline.py` lines 3-10: file-level code comments/docstrings provide code-level AI attribution.

**Status:** STRONG

---

### 9. Built multi-stage ML pipeline connecting outputs of one model to inputs of another (7 pts)

**Evidence:**
- `notebooks/reconstruction_v3.ipynb` cell 0 / lines 14-23 lays out the v3 chain: frame extraction -> Grounding DINO -> SAM 3 -> VGGT -> COLMAP-style dataset -> 2DGS -> TSDF -> fallback export.
- `notebooks/reconstruction_v2.ipynb` cell 17 / lines 776-789 documents Grounding DINO label `bottle container` becoming the richer SAM 3 prompt `green bottle container water bottle`.
- `notebooks/reconstruction_v3.ipynb` cell 17 / lines 1135-1137: RGBA PNGs are copied into the COLMAP image folder so 2DGS trains on SAM 3 alpha outputs.
- `notebooks/reconstruction_v3.ipynb` cell 17 / lines 1144-1157: VGGT outputs are converted into COLMAP-style camera metadata; intrinsics are scaled from 518-resolution inference to original image size.
- `notebooks/reconstruction_v4.ipynb` cell 0 / lines 16-21 lays out the final chain: frame selection -> SAM masks -> canonical frame -> SF3D / TripoSR -> validation/export.
- `src/pipeline.py` lines 111-235: SAM 3 output is converted into canonical RGBA input.
- `src/pipeline.py` lines 238-319: canonical RGBA output is converted into a TripoSR mesh input and OBJ output.

**Status:** STRONG

---

### 10. Performed error analysis with visualization and discussion of failure cases (7 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` cell 42 / lines 1466-1473: identifies TSDF failure root cause: white-background pixels with undefined/garbage depth are fused into real surfaces; records `45,944 disconnected clusters`.
- `notebooks/reconstruction_v2.ipynb` lines 1431-1435: Open3D debug output confirms `#clusters=45944`.
- `notebooks/reconstruction_v3.ipynb` cell 25: mesh quality gate reports `Connected components: 50`, largest component triangle ratio `0.842`, `NUMERIC_MESH_OK: False`, `VISUAL_TSDF_OK: False`, `MESH_OK: False`.
- `notebooks/reconstruction_v3.ipynb` cell 31: debugging checklist documents what to inspect next: RGBA images, alpha-mask loader, `mesh_utils`, VGGT intrinsics, and SuGaR fallback.
- `notebooks/reconstruction_v3.ipynb` cells 29-30: side-by-side 2DGS TSDF vs SuGaR visual comparison.
- `notebooks/reconstruction_v4.ipynb` cell 0 / lines 8-23: retrospective explains why v1/v2/v3 mesh extraction was unreliable and motivates the v4 pivot away from TSDF.
- `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png`: curated visualization comparing 2DGS, SuGaR, and v4 final mesh.

**Status:** STRONG

---

### 11. Compared multiple model architectures or approaches quantitatively with controlled experimental setup (7 pts)

**Evidence:**
- Same object/video is used across notebooks: `hydroflaskgreen.MOV` appears in v1, v2, v3, and v4.
- v1 approach: COLMAP + 3DGS + SuGaR. `notebooks/reconstruction_v1.ipynb` lines 6774-6778 records `L1 0.0021043826127424836`, `PSNR 35.61689376831055` at 7,000 iterations; lines 6903-6908 summarize `COLMAP -> 70/70`, `SAM 3`, and `35.6 dB PSNR`.
- v2/v3 approach: VGGT + 2DGS + TSDF. `notebooks/reconstruction_v2.ipynb` lines 1189-1193 records `L1 0.004935842612758279`, `PSNR 30.59425849914551` at 7,000 iterations; v3 quality gate records 50 connected components and failed visual approval.
- v4 approach: SAM 3 + TripoSR. `notebooks/reconstruction_v4.ipynb` lines 1290-1303 records `mesh.obj OK 3.9 MB`, selected final mesh, and saved final bundle.
- `README.md` lines 35-41 presents the comparison table: v1 `35.6 dB`, v2/v3 `30.6 dB`, v4 clean 3.9 MB OBJ.
- `notebooks/reconstruction_v3.ipynb` cells 29-30 and `assets/evaluation/2DGS_vs_SuGaR_vs_v4_final.png` provide qualitative visual comparison.

**Notes:** The comparison is controlled by using the same source object/video, but the approaches are not a strict hyperparameter-controlled benchmark. It is still strong evidence for comparing multiple architectures and approaches in the project context.

**Status:** STRONG

---

### 12. Documented at least two iterations of model improvement driven by evaluation results (5 pts)

**Evidence:**
- v1 -> v2: `notebooks/reconstruction_v2.ipynb` stage text describes replacing full COLMAP SfM with VGGT for speed; `notebooks/reconstruction_v2.ipynb` lines 861-864 says VGGT replaces the ~20 min COLMAP pipeline with a ~30 s forward pass.
- v2 -> v3: `notebooks/reconstruction_v3.ipynb` cell 0 / lines 10-23 states the v2 failure was white-background masks, so v3 keeps SAM masks as real alpha masks and trains 2DGS on RGBA PNGs.
- v3 -> v4: `notebooks/reconstruction_v4.ipynb` cell 0 / lines 8-23 states v1/v2/v3 splat-to-mesh outputs had floaters/fragments, so v4 switches to clean single-image mesh generation.
- `PIPELINE_SYSTEM_OVERVIEW.md` documents the versioned progression across v1-v4.
- `README.md` lines 43-49 includes the research progression and final v4 architecture diagrams.

**Status:** STRONG

---

### 13. Documented a design decision where you chose between ML approaches based on technical tradeoffs (3 pts)

**Evidence:**
- `notebooks/reconstruction_v2.ipynb` lines 861-867: documents VGGT replacing COLMAP for speed, with COLMAP fallback if VGGT export fails.
- `notebooks/reconstruction_v2.ipynb` lines 1023-1029: documents 2DGS replacing 3DGS + SuGaR to reduce runtime and add built-in TSDF extraction.
- `notebooks/reconstruction_v4.ipynb` cell 0 / lines 8-23: documents the decision to avoid TSDF mesh fusion from splats because prior outputs contained floaters/fragments.
- `notebooks/reconstruction_v4.ipynb` lines 581-589: documents Stable Fast 3D as preferred for coherent textured mesh assets from one image.
- `notebooks/reconstruction_v4.ipynb` lines 868-870: records SF3D install failure and decision to continue with TripoSR fallback.
- `notebooks/reconstruction_v4.ipynb` cell 21: explains the v4 tradeoff: cleaner mesh and less manual cleanup, but less exact multi-view photogrammetry fidelity.

**Status:** STRONG

---

### 14. Deployed model as functional web application with user interface (10 pts)

**Evidence:**
- `src/app.py` lines 29-30: creates FastAPI app and mounts `/outputs` static downloads.
- `src/app.py` lines 79-81: serves `static/index.html`.
- `src/app.py` lines 84-117: `/reconstruct/image` endpoint accepts uploaded image + prompt, creates a job, saves the image, and starts background reconstruction.
- `src/app.py` lines 120-126: `/jobs/{job_id}` endpoint exposes job status.
- `src/app.py` lines 60-68: completed job returns `mesh_url`, `preview_url`, `obj_url`, and `glb_url`.
- `static/index.html` lines 153-159: prompt input and drag/drop upload UI.
- `static/index.html` lines 164-168: `model-viewer` 3D preview plus `Download OBJ` and `Download GLB` links.
- `static/index.html` lines 200-215: frontend posts to `/reconstruct/image` and polls `/jobs/${jobId}`.
- `setup_colab.ipynb` cell 1 / lines 55-67: setup output shows `SAM 3 ready`, `TripoSR installed`, and `TrueRender demo setup complete`.
- `setup_colab.ipynb` cell 2 / lines 257-304: launches `uvicorn src.app:app`, starts `cloudflared tunnel`, extracts a `trycloudflare.com` public URL.
- `setup_colab.ipynb` cell 2 / lines 237-244: output includes a working `trycloudflare.com` URL.

**Notes:** The prompt mentions ngrok, but the current implementation uses Cloudflare Tunnel instead. The deployment evidence is still a public Colab tunnel.

**Status:** STRONG

---

### 15. Completed project individually without a partner (10 pts)

**Evidence:**
- `README.md` lines 65-69: Individual Contributions table lists Emmanuel Serrano as the sole contributor and describes all major work areas.
- Git evidence from `git shortlog -sn --all`: `94 Manny Serrano`.
- Git author evidence from `git log --format='%an <%ae>' | sort -u`: only Manny Serrano identities appear (`Manny Serrano <158786148+manny-serrano@users.noreply.github.com>` and `Manny Serrano <manny4285@gmail.com>`).

**Status:** STRONG

---

## Summary

| Claim | Points | Status |
|---|---:|---|
| 1. Modular code design | 3 | STRONG |
| 2. Data augmentation | 5 | WEAK |
| 3. Basic preprocessing | 3 | STRONG |
| 4. Substantive preprocessing pipeline | 7 | STRONG |
| 5. Vision transformer | 5 | STRONG |
| 6. Image segmentation model | 5 | STRONG |
| 7. Prompt engineering | 3 | WEAK |
| 8. AI attribution | 3 | STRONG |
| 9. Multi-stage ML pipeline | 7 | STRONG |
| 10. Error analysis | 7 | STRONG |
| 11. Model comparison | 7 | STRONG |
| 12. Iterative improvement | 5 | STRONG |
| 13. Design tradeoff | 3 | STRONG |
| 14. Web deployment | 10 | STRONG |
| 15. Individual project | 10 | STRONG |

**Maximum points represented by these claims:** 83

**Strongly supported points:** 75

**Weak / needs strengthening:** 8 points

### Items to strengthen before submission

- **Claim 2:** If the course staff interprets "data augmentation" strictly, the Laplacian sharpness filter may not count. Consider swapping this claim for a preprocessing/data-quality claim, or add a small true augmentation experiment with documented impact.
- **Claim 7:** The notebooks define three prompt candidates, but they stop once the first prompt succeeds. To make this STRONG, add a small table or notebook cell that runs all three prompts and records mask coverage / skipped frames / visual result for each.
- **README video links:** Not part of the 15 claims above, but the submission rubric still requires direct demo and technical walkthrough video links. `README.md` currently has TODO placeholders.
