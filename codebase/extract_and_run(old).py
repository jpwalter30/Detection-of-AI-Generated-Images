import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier


# -----------------------------
# Config
# -----------------------------
# Primary model (DINOv3 ViT-B/16). Fallback: DINOv2 for testing.
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # may require access approval on HF
# MODEL_NAME = "facebook/dinov2-base"                    # fallback if DINOv3 not yet available
TARGET_PCS = 500  # limit TabPFN to 500 dimensions

DATA_DIR = Path("data")
REAL_DIR = DATA_DIR / "real"
GEN_DIR  = DATA_DIR / "gen"


# -----------------------------
# Utilities
# -----------------------------
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (M-Chips) – MPS:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_images(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def get_image_paths() -> Tuple[List[Path], List[int]]:
    # Takes 2 images each: real (label 0) and generated (label 1)
    real_paths = sorted(list(REAL_DIR.glob("*.*")))[:2]
    gen_paths  = sorted(list(GEN_DIR.glob("*.*")))[:2]
    paths = real_paths + gen_paths
    labels = [0]*len(real_paths) + [1]*len(gen_paths)
    if len(paths) != 4:
        raise RuntimeError("Please place exactly 2 images each in data/real and data/gen.")
    return paths, labels


# -----------------------------
# DINOv3 Feature Extraction
# -----------------------------
@torch.no_grad()
def extract_features(images: List[Image.Image], model, processor, device) -> np.ndarray:
    """
    Returns an array of shape (N, D).
    - prefers outputs.pooler_output (CLS embedding),
    - fallback: first token from last_hidden_state.
    """
    feats = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output  # (1, D)
        else:
            emb = out.last_hidden_state[:, 0, :]  # (1, D)
        feats.append(emb.detach().cpu().numpy())
    return np.vstack(feats)  # (N, D)


# -----------------------------
# PCA to 500 + Standardization
# (fit only on "train" pair, transform on "test" pair)
# -----------------------------
def fit_pca_and_scale(X_train: np.ndarray, n_components: int = 500):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    Xp = pca.fit_transform(Xs)
    return scaler, pca, Xp


def transform_pca_and_scale(X_test: np.ndarray, scaler: StandardScaler, pca: PCA):
    Xs = scaler.transform(X_test)
    Xp = pca.transform(Xs)
    return Xp


# -----------------------------
# Main
# -----------------------------
def main():
    device = select_device()
    print(f"[Info] Using device: {device}")

    # 1) Data
    paths, labels = get_image_paths()
    images = load_images(paths)
    print("[Info] Images:", [p.name for p in paths])
    print("[Info] Labels:", labels, "(0=real, 1=generated)")

    # 2) Model
    print(f"[Info] Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device).eval()

    # 3) Features
    X = extract_features(images, model, processor, device)  # (4, D)
    y = np.array(labels, dtype=int)

    D = X.shape[1]
    print(f"[Info] Raw feature dimension: {D}")

    # Split: 1 real + 1 gen as "train context", 1 real + 1 gen as "test"
    # Assumption: [real1, real2, gen1, gen2]
    X_train = np.vstack([X[0], X[2]])  # 1 real, 1 gen
    y_train = np.array([0, 1], dtype=int)
    X_test  = np.vstack([X[1], X[3]])  # rest (1 real, 1 gen)
    y_test  = np.array([0, 1], dtype=int)

    # 4) PCA to 500 (fit on train), transform test
    scaler, pca, X_train_500 = fit_pca_and_scale(X_train, n_components=TARGET_PCS)
    X_test_500 = transform_pca_and_scale(X_test, scaler, pca)
    print(f"[Info] Train shape after PCA: {X_train_500.shape}, Test shape: {X_test_500.shape}")

    # 5) TabPFN: fit (very fast) and predict
    clf = TabPFNClassifier(device="cpu")  # CPU is sufficient locally; GPU later: device="cuda"
    clf.fit(X_train_500, y_train)
    y_pred = clf.predict(X_test_500)
    y_proba = clf.predict_proba(X_test_500)[:, 1]

    # 6) Results
    for i, (p, prob) in enumerate(zip(y_pred, y_proba)):
        fname = paths[[1, 3][i]].name  # corresponds to X_test order
        print(f"[Pred] {fname}: predicted={p} (prob_AI={prob:.3f}), true={y_test[i]}")

    # Optional: save features
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    np.save(out_dir/"X_raw.npy", X)
    np.save(out_dir/"y.npy", y)
    print("[Info] Features saved in ./outputs/")

if __name__ == "__main__":
    main()