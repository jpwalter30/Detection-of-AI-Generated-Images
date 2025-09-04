# Image Feature Extraction (DINOv3) with Tiny TabPFN Demo

This project provides small, focused scripts to extract global image features using DINOv3 ViT via Hugging Face `transformers`.

Optionally, a tiny TabPFN demo (2-shot train, 2-shot test) can be run on a toy dataset laid out as `data/real` and `data/gen`.


**What’s Inside**
- `extract_dinov3_features.py`: Extract CLS features from a DINOv3 ViT-S/16 model using Hugging Face `transformers`. Optional `--run_demo` uses TabPFN with a minimal split.
- `extract_and_run.py`: End-to-end minimal example using Hugging Face `facebook/dinov3-vitb16-pretrain-lvd1689m` to extract features, apply PCA→500, and run TabPFN.


**Requirements**
- Python 3.9–3.11 recommended
- PyTorch compatible with your system (CUDA or Apple MPS optional)
- Install deps:
  - `pip install -r requirements.txt`
  - If you plan to run `extract_and_run.py`, also install: `pip install transformers`

Notes:
- For the Hugging Face DINOv3 model (`facebook/dinov3-vitb16-pretrain-lvd1689m`), you may need to accept the model license on huggingface.co with your account before pulling weights.


**Project Structure**
- `data/`: Input images. For demos, use the two-folder layout below.
  - `data/real/`: Real images
  - `data/gen/`: Generated images
- `outputs/`: Saved feature arrays and metadata


**Quick Start**
- Create a virtual environment and install dependencies:
  - `python -m venv .venv && source .venv/bin/activate` (on macOS/Linux)
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt`
  - `pip install transformers`  (needed for `extract_and_run.py`)

- Prepare a tiny demo dataset:
  - Place at least two images per class:
    - `data/real/real1.jpeg`, `data/real/real2.jpeg`
    - `data/gen/gen1.jpg`, `data/gen/gen2.jpg`

**Workflow**

1. Prepare your data by placing images into the `data/real` and `data/gen` directories, with at least two images per class for the demo.

2. Run feature extraction using your chosen backbone:
   - For Hugging Face DINOv3 (recommended for ease of use), run:
     ```
     python extract_dinov3_features.py --images_dir ./data --hf_model facebook/dinov3-vitb16-pretrain-lvd1689m --out_dir ./outputs_dinov3 --run_demo
     ```

3. The extracted features, image paths, and optionally labels will be saved to the specified output directory.

4. Optionally, the `--run_demo` flag triggers PCA reduction to 500 dimensions and runs a tiny TabPFN demo (2-shot train/test) to verify the end-to-end pipeline.

5. Use the saved features for larger experiments such as GenImage analysis or leave-one-generator-out evaluations.


**Usage: End-to-End (Hugging Face Transformers)**
- Description:
  - `extract_and_run.py` uses `transformers.AutoImageProcessor` and `AutoModel` with `facebook/dinov3-vitb16-pretrain-lvd1689m` (default) to compute features for exactly 2 real and 2 generated images, applies PCA→500, and runs TabPFN.
- Run:
  - `python extract_and_run.py`
- Expectations:
  - Requires `data/real/` and `data/gen/` to contain exactly two images each; otherwise it raises an error.
  - Saves raw features to `outputs/` (`X_raw.npy`, `y.npy`).
  - If the DINOv3 HF weights are restricted, accept the license on the model card.


**Outputs**
- `X_cls.npy`: `float32` array of shape `(N, D)`; `D` depends on the backbone (e.g., ~384 for ViT-S/16).
- `paths.npy`: `object` array of file path strings (length `N`).
- `y.npy`: `int64` labels (0=real, 1=gen) when using the two-folder layout.
- `outputs/` in `extract_and_run.py` additionally stores `X_raw.npy` with raw (pre-PCA) embeddings.


**Devices**
- Scripts auto-select the best available device:
  - CUDA GPU if available → `cuda`
  - Apple Silicon → `mps`
  - Otherwise CPU → `cpu`
- You can force CPU in the TabPFN demo by leaving its `device` at `"cpu"` in the code.


**Troubleshooting**
- Missing `transformers`: Install with `pip install transformers` for `extract_and_run.py`.
- Hugging Face model access: Accept the `facebook/dinov3-vitb16-pretrain-lvd1689m` license on the model page when using the HF variant.
- MPS on macOS: Requires recent PyTorch builds; if you hit backend issues, switch device to CPU.


**Notes**
- The tiny TabPFN demo is intended only as a sanity check. For meaningful results, use realistic train/test sizes and splits.
- Feature dimensionality and performance can vary with image resolution. Both scripts default to 224×224 crops with ImageNet normalization.


**License**
- This repository bundles or references third-party models and code (DINOv3, Hugging Face Transformers, TabPFN). Please respect their respective licenses. No license is asserted for your own scripts unless you add one.


python extract_dinov3_features.py \--images_dir ./data \--hf_model facebook/dinov3-vitb16-pretrain-lvd1689m \--out_dir ./outputs_dinov3