"""Tiles, cells and folds of one processed cohort, served in CELLO's batch format.

A cohort directory $CELLPLEX_DATA/{cohort} (written by scripts/preprocess/build_dataset.py) holds
    meta.json            proteins, mpp, reader, image path template ('{sample}')
    cells/{sample}.npz   cx, cy (float32, native px, sorted by tile), y (n, P) float32 log1p mean intensity
                         in the cell region, cell_id (index in the source cell table)
    tiles.csv            sample, x0, y0, c0, c1 (cell range in cells/{sample}.npz), unit (split unit)
    folds.csv            unit, fold (0..4), as data/splits/{cohort}.csv
Fold k: test = units of fold k; val = the first half (rounded down, >= 1; fixed unit order of folds.csv) of the units of
fold (k + 1) % 5 (checkpoint selection only); train = all other units.
Items follow CELLO's PatchDataset: img (3, 224, 224) in [0, 1], cell_xy tile-local px (x, y), genes_target,
S_genes (= proteins), n_cells, patient (= sample).
"""
import json, os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .cello_utils import augment_image_and_positions, collate_fn, is_foreground_tile  # noqa: F401
from .cohorts import N_FOLDS, TILE
from .paths import cohort_dir


def load_meta(cohort):
    return json.load(open(f'{cohort_dir(cohort)}/meta.json'))


def load_tiles(cohort):
    return pd.read_csv(f'{cohort_dir(cohort)}/tiles.csv', dtype={'unit': str, 'sample': str})


def fold_units(cohort, fold, n_folds=N_FOLDS):
    """{'train', 'val', 'test'} -> set of units."""
    F = pd.read_csv(f'{cohort_dir(cohort)}/folds.csv', dtype={'unit': str})
    test = list(F.unit[F.fold == fold])
    nxt = list(F.unit[F.fold == (fold + 1) % n_folds])
    val = nxt[:max(1, len(nxt) // 2)]
    train = [u for u in F.unit if u not in set(test) | set(val)]
    return {'train': set(train), 'val': set(val), 'test': set(test)}


class Reader:
    """H&E tile reader: pyramidal TIFF via OpenSlide (level 0) or a uint8 (H, W, 3) .npy; outside -> white.
    $CELLPLEX_WSI_DIR, if set, replaces the directory of the WSIs (e.g. a copy on fast local disk)."""

    def __init__(self, meta):
        self.meta, self.h = meta, {}

    def path(self, s):
        return self.meta['images'][s] if self.meta.get('images') else self.meta['image'].format(sample=s)

    def _open(self, s):
        if s not in self.h:
            p = self.path(s)
            if self.meta['reader'] == 'wsi':
                import openslide
                d = os.environ.get('CELLPLEX_WSI_DIR')
                self.h[s] = openslide.OpenSlide(f'{d}/{os.path.basename(p)}' if d else p)
            else:
                self.h[s] = np.load(p, mmap_mode='r')
        return self.h[s]

    def size(self, s):
        """(W, H) at level 0"""
        h = self._open(s)
        return h.level_dimensions[0] if self.meta['reader'] == 'wsi' else (h.shape[1], h.shape[0])

    def read(self, s, x0, y0, T=TILE, h=None):
        """(h or T, T, 3) uint8 crop with its top-left corner at (x0, y0); pixels outside the image are white"""
        W_, H_ = T, h or T
        im = self._open(s)
        if self.meta['reader'] == 'wsi':
            a = np.asarray(im.read_region((int(x0), int(y0)), 0, (W_, H_)))
            img = a[..., :3].copy()
            img[a[..., 3] == 0] = 255
            return img
        H, W = im.shape[:2]
        img = np.full((H_, W_, 3), 255, np.uint8)
        ya, xa = max(int(y0), 0), max(int(x0), 0)
        yb, xb = min(int(y0) + H_, H), min(int(x0) + W_, W)
        if yb > ya and xb > xa:
            img[ya - int(y0):yb - int(y0), xa - int(x0):xb - int(x0)] = im[ya:yb, xa:xb]
        return img


class TileDS(Dataset):
    def __init__(self, cohort, tiles, max_cells_per_patch=256, augment=False):
        self.cohort = cohort
        self.meta = load_meta(cohort)
        self.proteins = list(self.meta['proteins'])
        self.tiles = tiles.reset_index(drop=True)
        self.cells = {s: dict(np.load(f'{cohort_dir(cohort)}/cells/{s}.npz')) for s in self.tiles['sample'].unique()}
        self.reader = Reader(self.meta)
        self.max_cells_per_patch, self.augment = max_cells_per_patch, augment
        self.patients = self.tiles['sample'].astype(str).values               # CELLO's per-slide grouping key
        self.S = self.tiles['sample'].values
        self.X0, self.Y0 = self.tiles.x0.values, self.tiles.y0.values
        self.C0, self.C1 = self.tiles.c0.values, self.tiles.c1.values

    def __len__(self):
        return len(self.tiles)

    def image(self, i):
        return self.reader.read(self.S[i], self.X0[i], self.Y0[i])

    def __getitem__(self, i):
        s, x0, y0, c0, c1 = self.S[i], self.X0[i], self.Y0[i], self.C0[i], self.C1[i]
        img = torch.from_numpy(self.image(i).astype(np.float32).transpose(2, 0, 1) / 255.)
        C = self.cells[s]
        coords = np.stack([C['cx'][c0:c1] - x0, C['cy'][c0:c1] - y0], 1).astype(np.float32)
        X = C['y'][c0:c1].astype(np.float32)
        idx = np.arange(c0, c1)
        n = len(coords)
        if self.max_cells_per_patch is not None and n > self.max_cells_per_patch:
            sel = np.random.choice(n, self.max_cells_per_patch, replace=False)
            coords, X, idx, n = coords[sel], X[sel], idx[sel], self.max_cells_per_patch
        rel_xy = torch.from_numpy(coords)
        if self.augment:
            img, rel_xy = augment_image_and_positions(img, rel_xy, TILE)
        return {'img': img, 'cell_xy': rel_xy, 'genes_target': torch.from_numpy(X), 'S_genes': self.proteins,
                'n_cells': n, 'patient': s, 'cell_idx': torch.from_numpy(idx)}


def split_datasets(cohort, fold, max_cells=256):
    T = load_tiles(cohort)
    U = fold_units(cohort, fold)
    return {k: TileDS(cohort, T[T.unit.isin(U[k])], max_cells_per_patch=max_cells if k == 'train' else None,
                      augment=(k == 'train')) for k in ('train', 'val', 'test')}, U
