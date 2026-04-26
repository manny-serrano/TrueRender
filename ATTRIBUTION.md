# Attribution

## AI Development Tools

I used Cursor's AI chat/agent as the main development assistant for this project. I used it to draft Colab notebook cells, translate research repo instructions into runnable setup cells, design fallback paths, and diagnose runtime errors from Colab package changes. I also used general ChatGPT/Claude-style prompting through Cursor for explanations of unfamiliar APIs such as VGGT's camera export, SAM 3 video prompting, and the 2DGS/TSDF mesh path. I did not treat the AI output as final code: the notebooks show repeated manual edits, probes, assertions, and patches after generated cells failed in the actual runtime.

## What Was AI-Generated

Substantial AI-assisted code appears throughout the reconstruction notebooks. In `reconstruction_v2.ipynb`, the frame filtering, Grounding DINO object selection, SAM 3 prompt retry loop, VGGT-to-COLMAP export, 2DGS training setup, and TSDF/Poisson mesh export cells were AI-drafted and then iterated. Specific examples include `reconstruction_v2.ipynb` cell 22, which rewrites VGGT intrinsics from 518 x 518 model coordinates to the original 1080 x 1920 image size, and cells 43-44, which patch 2DGS mesh extraction and add a Poisson fallback.

In `reconstruction_v3.ipynb`, AI helped generate the alpha-mask pipeline: saving SAM masks as real RGBA PNGs, building a COLMAP-style dataset whose image names point to `.png` files, installing/training 2DGS, and adding a mesh quality gate before export. `reconstruction_v3.ipynb` cell 9 contains the Grounding DINO setup, Transformers compatibility probes, and weight download logic. `reconstruction_v3.ipynb` cells 19 and 21 contain the Matplotlib/2DGS patch and training retry logic. In `reconstruction_v4.ipynb`, AI helped generate the cleaner image-to-3D pipeline: dependency baseline, SAM-based canonical frame selection, Stable Fast 3D attempt, TripoSR fallback, and packaging of final assets.

## What Was Modified

The AI-generated pipeline changed substantially after testing. The v2 white-background masking approach was replaced in v3 with real alpha masks because white JPG backgrounds confused later geometry steps. The VGGT export was rewritten so camera intrinsics were scaled back to the original frame resolution instead of leaving COLMAP metadata at 518 x 518. The v4 design was also a major rework: after v2/v3 produced fragmented splat-derived meshes, the project stopped treating TSDF as the final path and instead used the video only to select a clean SAM-masked single image for TripoSR.

Several generated setup cells were made idempotent and testable. For example, the 2DGS Matplotlib patch checks for the original `tostring_rgb` code, a previous bad patch, and an already-patched `buffer_rgba` version. The TripoSR setup cell restores `run.py` and `tsr/utils.py` before patching, then runs a child Python import probe.

## What Was Debugged or Reworked

The biggest debugging work was dependency and API drift. `reconstruction_v3.ipynb` cell 9 documents the Transformers pin saga: Grounding DINO expected `BertModel.get_head_mask`, while newer Transformers removed or changed that path. The final cell does a three-layer defense: force-pins `transformers==4.41.2`, checks the imported version in a child process, and patches `groundingdino/models/GroundingDINO/bertwarper.py` if needed.

In `reconstruction_v3.ipynb` cell 19, 2DGS failed because newer Matplotlib removed `FigureCanvasAgg.tostring_rgb()`. The fix changed the code to `fig.canvas.buffer_rgba()` and later corrected an AI-suggested reshape mistake in cell 21. In `reconstruction_v2.ipynb` cell 43, TSDF fusion produced tens of thousands of disconnected clusters because white background pixels contributed garbage depth; the patch masked near-white pixels in `mesh_utils.py`. When TSDF still produced fragments, `reconstruction_v2.ipynb` cell 44 added Screened Poisson reconstruction directly from opacity-filtered 2DGS Gaussian centers.

`reconstruction_v4.ipynb` also needed dependency repair. The first code cell pins `numpy==1.26.4` and restarts the kernel if NumPy was already imported with an incompatible ABI. Stable Fast 3D was attempted but failed on Colab/Python 3.12 compiled dependencies, so the notebook continued with TripoSR. `reconstruction_v4.ipynb` cell 16 patches both `run.py` and `tsr/utils.py` so `rembg` is imported lazily only when background removal is requested; this was needed because the project already supplies SAM-masked input with `--no-remove-bg`.

## Third-Party Code and Models

- Grounding DINO (`IDEA-Research/GroundingDINO`): text-prompted object detection.
- SAM 3 (`facebookresearch/sam3`): text-prompted video segmentation and mask propagation.
- VGGT (`facebookresearch/vggt`): feed-forward camera pose and 3D point estimation.
- 3D Gaussian Splatting (`graphdeco-inria/gaussian-splatting`): neural scene representation used in v1.
- 2D Gaussian Splatting (`hbb1/2d-gaussian-splatting`): surface-aligned splat training and TSDF extraction.
- SuGaR (`Anttwo/SuGaR`): surface-aligned mesh extraction from 3DGS.
- TripoSR (`VAST-AI-Research/TripoSR`): single-image-to-3D mesh generation in v4.
- Stable Fast 3D (`Stability-AI/stable-fast-3d`): alternative image-to-3D path attempted in v4.
- COLMAP: structure-from-motion and COLMAP-format dataset conventions.
- Hugging Face Hub: model weights including `facebook/VGGT-1B` and `ShilongLiu/GroundingDINO`.

## Datasets

The project uses my own phone video, `hydroflaskgreen.MOV`, as the source dataset. The frames, masks, RGBA training images, COLMAP exports, splat checkpoints, and final meshes are derived from that video; no external training dataset was added by me.
