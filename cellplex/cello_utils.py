"""Helpers from CELLO (arXiv:2609.36429, cello/dataset.py and cello/metrics.py), unchanged: training
augmentation, background filter, batch collation and the per-sample protein PCC used for checkpoint selection."""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr


def augment_image_and_positions(img: torch.Tensor, rel_xy: torch.Tensor, size: int):
    """Random horizontal / vertical flips (cell coordinates flipped accordingly)
    and mild brightness / contrast jitter. Training split only."""
    if random.random() < 0.5:
        img = TF.hflip(img)
        rel_xy = rel_xy.clone()
        rel_xy[:, 0] = (size - 1) - rel_xy[:, 0]
    if random.random() < 0.5:
        img = TF.vflip(img)
        rel_xy = rel_xy.clone()
        rel_xy[:, 1] = (size - 1) - rel_xy[:, 1]
    if random.random() < 0.5:
        pil = TF.to_pil_image(img.clamp(0, 1))
        pil = TF.adjust_brightness(pil, 0.9 + 0.2 * random.random())
        pil = TF.adjust_contrast(pil, 0.9 + 0.2 * random.random())
        img = TF.to_tensor(pil)
    return img, rel_xy


def is_foreground_tile(img_uint8: np.ndarray, lo=10, hi=250, min_std=5) -> bool:
    """Background filter used by whole-slide validation (mean in [lo, hi], std >= min_std)."""
    return lo <= img_uint8.mean() <= hi and img_uint8.std() >= min_std


def collate_fn(batch):
    """Pad a variable number of cells per tile; ``cell_mask`` marks real cells."""
    imgs = torch.stack([b["img"] for b in batch])
    n_cells = [b["n_cells"] for b in batch]
    max_n = max(n_cells)
    B = len(batch)
    n_genes = batch[0]["genes_target"].shape[-1]

    cell_xy_pad = torch.zeros(B, max_n, 2)
    genes_target_pad = torch.zeros(B, max_n, n_genes)
    cell_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["n_cells"]
        cell_xy_pad[i, :n] = b["cell_xy"][:n]
        genes_target_pad[i, :n] = b["genes_target"][:n]
        cell_mask[i, :n] = True

    return {
        "img": imgs,
        "cell_xy": cell_xy_pad,
        "genes_target": genes_target_pad,
        "cell_mask": cell_mask,
        "n_cells": n_cells,
        "S_genes": batch[0]["S_genes"],
        "patient": [b["patient"] for b in batch],
    }


def slide_gene_pcc(preds, gts):
    """Mean gene-wise PCC used for checkpoint selection during training.

    Genes whose ground truth or prediction is constant yield NaN and are skipped.
    """
    vals = []
    with np.errstate(all="ignore"):
        for g in range(preds.shape[1]):
            x, y = gts[:, g], preds[:, g]
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() >= 2:
                if np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    continue
                r, _ = pearsonr(x[m], y[m])
                if not np.isnan(r):
                    vals.append(r)
    return float(np.mean(vals)) if vals else 0.0
