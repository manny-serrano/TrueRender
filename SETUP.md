# Setup Guidelines

The recommended way to run TrueRender is through `setup_colab.ipynb`. The demo depends on GPU packages, SAM 3 model access, TripoSR, and a temporary public tunnel, so Colab is the most reliable setup path for graders and reviewers.

## Requirements

- Google Colab with an A100 GPU runtime preferred.
- A Hugging Face account.
- Access accepted for [`facebook/sam3`](https://huggingface.co/facebook/sam3).
- A Hugging Face read token from [`huggingface.co/settings/tokens`](https://huggingface.co/settings/tokens).
- A JPG or PNG image containing a single object to reconstruct.

## Step-by-Step Colab Setup

1. Open `setup_colab.ipynb` in Google Colab.

2. Set the runtime to GPU:
   `Runtime` -> `Change runtime type` -> `Hardware accelerator` -> `A100 GPU` if available.

3. Run **Cell 0** and paste your Hugging Face token when prompted. This stores `HF_TOKEN` in the Colab environment for the rest of the notebook.

4. Run **Cell 1**. This cell installs the TrueRender demo dependencies, logs in to Hugging Face, clones and installs SAM 3, applies the SAM 3 compatibility patch, clones and installs TripoSR, patches TripoSR so it can use the SAM-masked input without `rembg`, and verifies the imports.

5. Wait for Cell 1 to print:
   ```text
   TrueRender demo setup complete
   ```

6. Run **Cell 2**. This starts the FastAPI app with `uvicorn` and creates a temporary Cloudflare tunnel.

7. Copy the printed `trycloudflare.com` URL and open it in a browser.

8. In the web UI, enter a text prompt for the object, such as `shoe`, `bottle`, or `cup`, then upload a JPG or PNG image.

9. Wait for the pipeline to finish. The UI will show progress through segmentation, crop preparation, mesh generation, and final preview.

10. Download the generated `OBJ` or `GLB` mesh from the result page.

## What the Notebook Installs

`setup_colab.ipynb` installs and configures:

- FastAPI and Uvicorn for the web demo.
- SAM 3 for text-guided object segmentation.
- TripoSR for single-image mesh generation.
- Cloudflare tunnel support so the Colab-hosted app can be opened in a browser.
- Compatibility patches needed for the current Colab Python/package environment.

## Troubleshooting

- If Cell 0 says the token looks wrong, confirm the token starts with `hf_` and has read access.
- If SAM 3 fails to download, make sure you accepted the model terms on Hugging Face before running the notebook.
- If Cell 1 fails partway through installation, restart the Colab runtime and rerun from Cell 0.
- If the Cloudflare URL does not load immediately, wait 10-20 seconds and refresh the page.
- If a reconstruction fails, try a clearer image with one centered object and use a simple prompt matching the object.
