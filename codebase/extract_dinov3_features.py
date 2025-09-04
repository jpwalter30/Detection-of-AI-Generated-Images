"""
Extract CLS features from images using a Hugging Face DINOv3 model.

Usage:
  # Mode 1: Batch feature extraction (default)
  python extract_dinov3_features.py \
      --images_dir data \
      --hf_model facebook/dinov3_vits16 \
      --out_dir outputs

  # Mode 2: Tiny 2-shot TabPFN demo (requires at least 2 real/ & 2 gen/ images in data/real and data/gen)
  python extract_dinov3_features.py \
      --images_dir data \
      --hf_model facebook/dinov3_vits16 \
      --mode demo2
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import json

# optional for the tiny demo
try:
    from tabpfn import TabPFNClassifier
    TABPFN_OK = True
except Exception:
    TABPFN_OK = False

# transformers for Hugging Face DINOv3
try:
    from transformers import AutoImageProcessor, AutoModel
    HF_TRANSFORMERS_OK = True
except ImportError:
    HF_TRANSFORMERS_OK = False


# ------------------------- helpers -------------------------
def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def list_images_two_folders(root: Path) -> Tuple[List[Path], List[int]]:
    real_dir = root / "real"
    gen_dir = root / "gen"
    real = list_images(real_dir) if real_dir.exists() else []
    gen = list_images(gen_dir) if gen_dir.exists() else []
    paths = real + gen
    labels = [0] * len(real) + [1] * len(gen)
    return paths, labels


def load_image(path: Path) -> Image.Image:
    # convert to RGB and ignore EXIF orientation (handled by PIL)
    return Image.open(path).convert("RGB")




def fit_pca_500(X_train: np.ndarray) -> Tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    n_comp = min(500, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(Xs)
    return scaler, pca, Xp


def transform_pca_500(X: np.ndarray, scaler: StandardScaler, pca: PCA) -> np.ndarray:
    return pca.transform(scaler.transform(X))


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="Folder with images or subfolders real/ and gen/")
    ap.add_argument("--hf_model", type=str, required=True, help="Hugging Face model name (e.g., facebook/dinov3_vits16)")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--mode", type=str, choices=["extract", "demo2"], default="extract",
                    help="extract = dump features for all images; demo2 = tiny 2-shot TabPFN PoC.")
    args = ap.parse_args()

    global hf_model_name
    hf_model_name = args.hf_model

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_auto()
    print(f"[INFO] Using device: {device}")

    if not HF_TRANSFORMERS_OK:
        raise SystemExit("[ERROR] transformers package not installed, cannot load Hugging Face model.")
    print(f"[INFO] Loading Hugging Face model: {hf_model_name}")
    processor = AutoImageProcessor.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)
    model.to(device)
    model.eval()

    # 3) Collect images
    #    (a) try two-folder layout (real/ + gen/) for labels
    paths, labels = list_images_two_folders(images_dir)
    if len(paths) == 0:
        #    (b) otherwise flat directory (no labels)
        paths = list_images(images_dir)
        labels = None

    if len(paths) == 0:
        raise SystemExit(f"No images found under {images_dir}")

    print(f"[INFO] Found {len(paths)} images.")
    if args.mode == "extract":
        # 4) Extract CLS features in batches
        X_list = []
        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i:i+args.batch_size]
            pil_imgs = [load_image(p) for p in batch_paths]
            inputs = processor(images=pil_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            z = outputs.last_hidden_state[:, 0, :]
            X_list.append(z.detach().cpu().numpy())
        X = np.concatenate(X_list, axis=0)
        print(f"[INFO] Feature matrix shape: {X.shape}")

        # 5) Save features
        np.save(out_dir / "X_cls.npy", X)
        np.save(out_dir / "paths.npy", np.array([str(p) for p in paths], dtype=object))
        if labels is not None:
            np.save(out_dir / "y.npy", np.array(labels, dtype=np.int64))
        print(f"[INFO] Saved to {out_dir}/(X_cls.npy, paths.npy{', y.npy' if labels is not None else ''})")

    elif args.mode == "demo2":
        if not TABPFN_OK:
            print("[WARN] tabpfn not installed; skipping demo.")
            return
        # Need at least 2 real + 2 gen images, with labels
        if labels is None:
            print("[WARN] Need labeled data in real/ and gen/ subfolders for demo2 mode.")
            return
        real_idx = [i for i, y in enumerate(labels) if y == 0][:2]
        gen_idx  = [i for i, y in enumerate(labels) if y == 1][:2]
        if len(real_idx) < 2 or len(gen_idx) < 2:
            print("[WARN] Need at least 2 real and 2 gen images under images_dir/real and images_dir/gen for demo2 mode.")
            return
        sel_indices = [real_idx[0], real_idx[1], gen_idx[0], gen_idx[1]]
        sel_paths = [paths[i] for i in sel_indices]
        sel_labels = [labels[i] for i in sel_indices]

        # Extract features for only these 4 images (order: real1, real2, gen1, gen2)
        pil_imgs = [load_image(p) for p in sel_paths]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        X_sel = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

        # Train: real1 + gen1; Test: real2 + gen2
        X_train = np.vstack([X_sel[0], X_sel[2]])
        y_train = np.array([0, 1], dtype=int)
        X_test  = np.vstack([X_sel[1], X_sel[3]])
        y_test  = np.array([0, 1], dtype=int)

        scaler, pca, Xtr = fit_pca_500(X_train)
        Xte = transform_pca_500(X_test, scaler, pca)

        clf = TabPFNClassifier(device="cpu")  # set "cuda" on GPU box
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_proba = clf.predict_proba(Xte)[:, 1]

        test_names = [sel_paths[1].name, sel_paths[3].name]
        for name, yp, pr, yt in zip(test_names, y_pred, y_proba, y_test):
            print(f"[DEMO] {name}: pred={yp} (prob_AI={pr:.3f}), true={yt}")

        # ---- Demo metrics (tiny/illustrative only) ----
        if len(y_test) >= 1:
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = float("nan")

            print("[DEMO][metrics] "
                  f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  roc_auc={auc}")

            # Save predictions + metrics
            metrics = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(auc) if isinstance(auc, float) else None,
                "n_test": int(len(y_test)),
                "test_names": test_names,
                "y_true":  [int(v) for v in y_test.tolist()],
                "y_pred":  [int(v) for v in y_pred.tolist()],
                "y_proba": [float(v) for v in y_proba.tolist()],
            }
            with open(out_dir / "demo_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Optional: save ROC curve (if matplotlib is available)
            try:
                import matplotlib.pyplot as plt
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                plt.figure()
                label = f"AUC={auc:.3f}" if isinstance(auc, float) else "ROC"
                plt.plot(fpr, tpr, label=label)
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Demo ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "demo_roc.png", dpi=150)
                plt.close()
            except Exception:
                pass  # matplotlib not installed? -> skip the plot

            if len(y_test) < 10:
                print("[DEMO][note] Very few test samples; metrics are only indicative.")

        print("[DEMO] Done.")


if __name__ == "__main__":
    main()