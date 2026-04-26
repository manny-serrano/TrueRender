# <img width="200" height="200" alt="Image" src="https://github.com/user-attachments/assets/2ce13084-fb7f-4dbb-b635-8e5bc0274ed3" /> TrueRender

## What it Does

TrueRender turns a single photo of an object into a browser-previewable 3D mesh. It uses SAM 3 to segment the object from the image based on a text prompt, crops a canonical RGBA frame, and feeds it to TripoSR for single-image 3D reconstruction. The result is served as a downloadable OBJ and GLB via a FastAPI web app. Earlier pipeline variants explored multi-view reconstruction (COLMAP + 3DGS + SuGaR) and faster alternatives (VGGT + 2DGS + TSDF); the final demo uses the SAM 3 + TripoSR path, which produces clean meshes in seconds.

---

## Quick Start

**Requirements:** Google Colab with an A100 GPU runtime and a [HuggingFace account](https://huggingface.co) with access to [`facebook/sam3`](https://huggingface.co/facebook/sam3).

1. Clone the repo and open `setup_colab.ipynb` in Colab.
2. Run **Cell 0** — paste your HuggingFace token when prompted.
3. Run **Cell 1** — installs all dependencies, patches SAM 3 and TripoSR, logs in to HuggingFace.
4. Run **Cell 2** — starts the FastAPI server and prints a public Cloudflare tunnel URL.
5. Open the URL, type a prompt (e.g. `shoe`, `bottle`, `cup`), and drop in a JPG or PNG.

The UI steps through segmenting → cropping → generating mesh → done, then shows a 3D preview with OBJ and GLB download links.

---

## Video Links

- **Demo video:** [link TBD]
- **Technical walkthrough:** [link TBD]

---

## Evaluation

| Pipeline | Reconstruction Quality | Runtime |
|---|---|---|
| v1: COLMAP + SAM 3 + 3DGS + SuGaR | PSNR 35.6 dB, L1 = 0.00210 at 7k iterations — usable SuGaR mesh | ~130 min total |
| v2/v3: VGGT + SAM 3 + 2DGS + TSDF | PSNR 30.6 dB, L1 = 0.00494 — failed quality gate (50 connected components) | ~12–30 min |
| v4: SAM 3 + TripoSR (final) | Clean single-component OBJ, 3.9 MB, passes geometry validation | Seconds after setup |

v1 produced usable geometry but required ~130 minutes of COLMAP + 3DGS + SuGaR processing. v2/v3 were faster but TSDF mesh extraction from 2DGS remained fragmented around masked regions. v4 avoids multi-view fusion entirely: SAM 3 isolates a clean canonical frame and TripoSR generates a coherent mesh from it, making it the practical choice for a live demo.

---

## Individual Contributions

| Contributor | Contributions |
|---|---|
| Emmanuel Serrano | Pipeline design and implementation (all four variants), SAM 3 integration, TripoSR integration, FastAPI demo app, Colab setup notebook, evaluation |
