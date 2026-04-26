# <img width="200" height="200" alt="Image" src="https://github.com/user-attachments/assets/2ce13084-fb7f-4dbb-b635-8e5bc0274ed3" /> TrueRender

TrueRender reconstructs an object captured in a casual phone video into a usable 3D mesh asset. The notebooks explore four pipeline variants: v1 is the COLMAP + SAM 3 + 3D Gaussian Splatting + SuGaR baseline, v2 and v3 replace the slow reconstruction steps with VGGT poses and 2DGS/TSDF mesh extraction, and v4 pivots to a clean image-to-3D path from a SAM-masked canonical frame. In practice, the v1 SuGaR pipeline and the v4 TripoSR pipeline produce usable meshes; the v2/v3 2DGS path trains quickly but produces fragmented meshes.



## System Overview

![Image](https://github.com/user-attachments/assets/f4dce161-1d19-45f5-a4dc-2fd7c3a762ac)


## Models Used
- COLMAP
- SAM 3 META
- 3D Gaussian Splatting
- SuGaR
- Grounding DINO
- VGGT
- 2D Gaussian Splatting
- Stable Fast 3D
- TripoSR

## What it Does

TrueRender takes a phone video of an object and turns it into a 3D mesh of the captured object. We explored four pipeline variants: v1 baseline reconstruction with COLMAP, SAM 3, 3DGS, and SuGaR; v2/v3 faster reconstruction with VGGT camera poses and 2DGS mesh extraction; and v4, an image-to-3D pivot that selects a clean SAM-masked object frame and generates a mesh from that image. The v1 SuGaR pipeline and v4 TripoSR pipeline produce usable meshes, while v2/v3 are useful for faster experimentation but did not pass the final mesh quality gate.

## Quick Start

Use `notebooks/reconstruction_v4.ipynb` as the recommended end-to-end path.

1. Open `notebooks/reconstruction_v4.ipynb` in Google Colab and select a GPU runtime.
2. Put the input video on Google Drive at `/content/drive/MyDrive/hydroflaskgreen.MOV`, matching the `VIDEO_PATH` used in the notebook.
3. Provide a Hugging Face token when prompted or set `HF_TOKEN` in Colab Secrets. This is required for SAM 3 downloads and for gated image-to-3D models such as Stable Fast 3D.
4. Run the v4 notebook from the top in order: runtime dependency baseline, global configuration and Drive mount, Stage 1 frame extraction, Stage 2 SAM 3 segmentation, Stage 3 canonical frame selection, Stage 4A Stable Fast 3D if available, Stage 4B TripoSR, then Stage 5 validation and packaging.
5. The validated v4 output is copied to `/content/drive/MyDrive/truerender_final_v4` with the final OBJ bundle and `source_manifest.json`.

For full multi-view reconstruction, use `notebooks/reconstruction.ipynb` as the v1 alternative. That path runs frame extraction, COLMAP SfM, SAM 3 masking, 3DGS training, and SuGaR mesh extraction, and is slower but produced a usable SuGaR mesh.

## Video Links

- Demo video: [link TBD]
- Technical walkthrough: [link TBD]

## Evaluation

| Pipeline | PSNR | Mesh quality | Runtime |
|---|---:|---|---|
| v1: COLMAP + SAM 3 + 3DGS + SuGaR | 35.6 dB, L1 = 0.00210 at 7000 iterations on 70 SAM-masked frames | Usable SuGaR mesh from the full multi-view pipeline | COLMAP SfM ~20 min; 3DGS + SuGaR ~110 min |
| v2/v3: VGGT + SAM 3 + 2DGS + TSDF | 30.59 dB, L1 = 0.00494 at 7000 iterations on 30 frames | Failed quality gate: 50 connected components, largest-component triangle ratio 0.842 | VGGT ~16-30s; 2DGS ~12 min |
| v4: SAM 3 canonical frame + TripoSR | Not a splat-training pipeline | Clean validated OBJ mesh, 3.9 MB, non-empty geometry, single connected component | TripoSR mesh generation completes in seconds after setup |

The frame filtering used variance-of-Laplacian sharpness scoring and kept the 30 sharpest frames from 70 candidates for the v2/v3 path: best sharpness 199.0, cutoff 48.9, and worst dropped frame 14.4. v1 succeeded because full COLMAP + 3DGS + SuGaR preserved enough multi-view geometry for a coherent object mesh, despite the long runtime. v4 succeeded by avoiding splat-to-TSDF fusion entirely: it uses SAM 3 to isolate a strong canonical object image and lets TripoSR generate a coherent object mesh. v2/v3 were much faster, but their TSDF mesh extraction from 2DGS output remained fragmented, especially around masked or low-opacity background regions.


