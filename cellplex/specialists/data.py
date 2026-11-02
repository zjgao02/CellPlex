"""Pixel-level training windows and cell-level predictions of the specialist models (Eva, GigaTIME).

Besides the files of cellplex/data.py, a processed cohort holds (scripts/preprocess/):
  array cohorts (reader 'npy'; cores / regions are predicted whole)
    pixel/{sample}.npz        t (P, H, W) uint8 multiplex target in the H&E frame (per sample and channel
                              p0.5-p99.5 over tissue -> [0, 255]), mask (H, W) tissue; or, for large samples,
                              pixel/{sample}_t.npy (H, W, P) and pixel/{sample}_mask.npy
    quant/{sample}.npz        inst (H, W) label image of the cell regions (nucleus + 2 um; label k + 1 = cell k of the
                              source cell table, the cell_id of cells/{sample}.npz)
  whole-slide cohorts (reader 'wsi'; predicted in 2048 px blocks with 256 px of context)
    pixel/{sample}_norm.npz   if_path (multiplex image, pyramidal TIFF (C, H, W)), channels (panel order), lo, hi
                              (per-channel scaling), low (coarse tissue mask of window centres), factor (its
                              downsampling), and A, field, tile when the multiplex image is registered to the H&E
                              (affine + local field) rather than on its pixel grid
    seg/{sample}_cells.json   CellViT++ nuclei (contour, centroid) in H&E level-0 px, in source cell order
A cell's prediction is the mean prediction over its nucleus + 2 um region (H&E frame).
"""
import json, os, time

import cv2
import numpy as np
import pandas as pd
import tifffile
import torch

from ..data import Reader, load_meta, load_tiles
from ..paths import cohort_dir

EXPAND_UM = 2.0


def tissue_of(he):
    return cv2.blur(cv2.cvtColor(np.ascontiguousarray(he), cv2.COLOR_RGB2HSV)[..., 1], (5, 5)) > 20


def hsv_jitter(img, rng, h=0.02, s=0.1, v=0.1):
    x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    x[..., 0] = (x[..., 0] + rng.uniform(-h, h) * 180) % 180
    x[..., 1] *= rng.uniform(1 - s, 1 + s)
    x[..., 2] *= rng.uniform(1 - v, 1 + v)
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def field_at(field, tile, cx, cy):
    """Local shift (multiplex px) at multiplex coordinates: bilinear between tile centres."""
    fx = (cx - tile / 2) / tile; fy = (cy - tile / 2) / tile
    return [cv2.remap(field[..., c].astype(np.float32), fx.astype(np.float32), fy.astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in range(2)]


def multiplex_on_he(z, reg, x0, y0, w, h, chans):
    """Registered multiplex channels sampled bilinearly on the H&E pixel grid of a window -> (len(chans), h, w)
    float32 and the mask of pixels that fall inside the multiplex image."""
    C, HC, WC = z.shape
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    A = reg['A']
    ax = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]; ay = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    dx, dy = field_at(reg['field'], int(reg['tile']), ax, ay)
    mx, my = ax + dx, ay + dy
    inside = (mx >= 0) & (my >= 0) & (mx <= WC - 1) & (my <= HC - 1)
    out = np.zeros((len(chans), h, w), np.float32)
    if not inside.any():
        return out, inside
    X0, Y0 = int(max(0, np.floor(mx[inside].min()) - 2)), int(max(0, np.floor(my[inside].min()) - 2))
    X1, Y1 = int(min(WC, np.ceil(mx[inside].max()) + 3)), int(min(HC, np.ceil(my[inside].max()) + 3))
    cod = np.asarray(z[:, Y0:Y1, X0:X1])
    u, v = (mx - X0).astype(np.float32), (my - Y0).astype(np.float32)
    for j, c in enumerate(chans):
        out[j] = cv2.remap(cod[c].astype(np.float32), u, v, cv2.INTER_LINEAR, borderValue=0)
    return out, inside


def load_nuclei(cohort, s):
    """(polygons, centroids) of a whole-slide sample in H&E level-0 px, in source cell order"""
    J = json.load(open(f'{cohort_dir(cohort)}/seg/{s}_cells.json'))
    return [np.asarray(c['contour'], np.float64) for c in J['cells']], np.array([c['centroid'] for c in J['cells']], np.float64)


def unit_samples(cohort, units):
    T = load_tiles(cohort)
    return sorted(T[T.unit.isin(set(units))]['sample'].unique())


def unit_cells(cohort, units):
    """{sample: rows of cells/{sample}.npz} of the cells in the tiles of the given split units"""
    T = load_tiles(cohort)
    T = T[T.unit.isin(set(units))]
    return {s: np.concatenate([np.arange(a, b) for a, b in zip(g.c0, g.c1)]) for s, g in T.groupby('sample')}


class Windows(torch.utils.data.Dataset):
    """Random field-of-view windows (L native px) centred on tissue pixels of the training samples; returns the H&E
    resized to W px, the native target crop (L, L, P) uint8 and its supervision mask (tissue). Array cohorts: targets
    in memory (or memory-mapped); whole-slide cohorts: H&E and multiplex crops read from the slides (opened lazily in
    each worker) and scaled with pixel/{sample}_norm.npz."""

    def __init__(self, cohort, units, n, L, W, seed=0, aug=True):
        d = cohort_dir(cohort)
        self.meta = load_meta(cohort)
        self.wsi = self.meta['reader'] == 'wsi'
        self.n, self.L, self.W, self.seed, self.aug = n, L, W, seed, aug
        self.h = {}
        samples = unit_samples(cohort, units)
        if self.wsi:
            self.N = {c: dict(np.load(f'{d}/pixel/{c}_norm.npz', allow_pickle=True)) for c in samples}
            self.hepath = {c: Reader(self.meta).path(c) for c in samples}
            self.pix = {c: (np.argwhere(self.N[c]['low']) * float(self.N[c]['factor'])).astype(np.int64) for c in samples}
        else:
            R = Reader(self.meta)
            self.he = {c: np.asarray(R._open(c)) for c in samples}
            if os.path.exists(f'{d}/pixel/{samples[0]}_t.npy'):           # large cohorts: memory-mapped (H, W, P)
                self.t = {c: np.load(f'{d}/pixel/{c}_t.npy', mmap_mode='r') for c in samples}
                self.m = {c: np.load(f'{d}/pixel/{c}_mask.npy') for c in samples}
            else:
                P = {c: np.load(f'{d}/pixel/{c}.npz') for c in samples}
                self.t = {c: np.ascontiguousarray(P[c]['t'].transpose(1, 2, 0)) for c in samples}
                self.m = {c: P[c]['mask'] for c in samples}
            self.pix = {c: np.argwhere(self.m[c]) for c in samples}
        self.samples = [c for c in sorted(self.pix) if len(self.pix[c])]
        w = np.array([len(self.pix[c]) for c in self.samples], np.float64)
        self.w = w / w.sum()

    def __len__(self):
        return self.n

    def _shape(self, c):
        return self._slides(c)[1].shape[:2] if self.wsi else self.m[c].shape

    def _slides(self, c):
        """(multiplex zarr (C, H, W), H&E zarr (H, W, 3+)) of a whole-slide sample, opened once per worker process"""
        if c not in self.h:
            import zarr
            self.h[c] = (zarr.open(tifffile.imread(str(self.N[c]['if_path']), aszarr=True, level=0), mode='r'),
                         zarr.open(tifffile.imread(self.hepath[c], aszarr=True, level=0), mode='r'))
        return self.h[c]

    def _crop(self, c, y0, x0, hh, ww):
        if self.wsi:
            z, he = self._slides(c)
            N = self.N[c]
            img = np.asarray(he[y0:y0 + hh, x0:x0 + ww])[..., :3]
            if 'A' in N:                                               # multiplex sampled through the registration
                a, inside = multiplex_on_he(z, N, x0, y0, ww, hh, N['channels'])
            else:                                                      # multiplex on the H&E grid
                a = np.asarray(z[:, y0:y0 + hh, x0:x0 + ww])[N['channels']].astype(np.float32)
                inside = np.ones((hh, ww), bool)
            lo, hi = N['lo'][:, None, None], N['hi'][:, None, None]
            t = np.round(np.clip((a - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            return img, t, (tissue_of(img) & inside).astype(np.uint8)
        return self.he[c][y0:y0 + hh, x0:x0 + ww], np.asarray(self.t[c][y0:y0 + hh, x0:x0 + ww]), self.m[c][y0:y0 + hh, x0:x0 + ww]

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        c = self.samples[rng.choice(len(self.samples), p=self.w)]
        H, Wd = self._shape(c)
        L = self.L
        cy, cx = self.pix[c][rng.integers(len(self.pix[c]))]
        if self.wsi:                                                  # centre on a coarse tissue cell: jitter inside it
            f = float(self.N[c]['factor'])
            cy, cx = cy + rng.uniform(0, f), cx + rng.uniform(0, f)
        y0 = int(np.clip(cy - L // 2, 0, max(0, H - L))); x0 = int(np.clip(cx - L // 2, 0, max(0, Wd - L)))
        hh, ww = min(L, H - y0), min(L, Wd - x0)
        a, b, k = self._crop(c, y0, x0, hh, ww)
        img = np.full((L, L, 3), 255, np.uint8); t = np.zeros((L, L, b.shape[2]), np.uint8); m = np.zeros((L, L), np.uint8)
        img[:hh, :ww] = a; t[:hh, :ww] = b; m[:hh, :ww] = k
        img = cv2.resize(img, (self.W, self.W), interpolation=cv2.INTER_AREA if L > self.W else cv2.INTER_LINEAR)
        if self.aug:
            k = int(rng.integers(4)); fl = rng.random() < 0.5
            op = lambda v: (np.rot90(v, k)[:, ::-1] if fl else np.rot90(v, k))   # noqa: E731
            img, t, m = op(img), op(t), op(m)
            img = hsv_jitter(np.ascontiguousarray(img), rng)
        return (torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(t)),
                torch.from_numpy(np.ascontiguousarray(m)))


def _accumulate(S, Cn, p, lab):
    K = p.shape[0]
    S.index_add_(0, lab, p.reshape(K, -1).T.double())
    Cn.index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))


def predict_cells(model, cohort, units, dev, block=2048, margin=256, max_blocks=None, log=print):
    """{sample: (cell_idx, pred, truth)} for the cells of the units (cells/{sample}.npz order); model.predict_image(he)
    -> (P, H, W) maps at native resolution. Array samples are predicted whole, slides in blocks with context
    (max_blocks: only a fixed random subset of the blocks, and only their cells, for validation)."""
    d = cohort_dir(cohort)
    meta = load_meta(cohort)
    reader = Reader(meta)
    out = {}
    for s, sel in unit_cells(cohort, units).items():
        B = np.load(f'{d}/cells/{s}.npz')
        cid = B['cell_id'][sel]
        if meta['reader'] != 'wsi':
            Z = np.load(f'{d}/quant/{s}.npz')
            inst = Z['inst']
            n = len(Z['cx']) if 'cx' in Z else int(inst.max())
            p = model.predict_image(np.asarray(reader._open(s)))
            S = torch.zeros(n + 1, p.shape[0], dtype=torch.float64, device=dev); Cn = torch.zeros(n + 1, dtype=torch.float64, device=dev)
            _accumulate(S, Cn, p, torch.from_numpy(inst.astype(np.int64)).to(dev).reshape(-1))
        else:
            from skimage.segmentation import expand_labels
            e = EXPAND_UM / meta['mpp']
            polys, cen = load_nuclei(cohort, s)
            W, H = reader.size(s)
            bb = np.array([[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()] for q in polys])
            n = len(polys)

            def labels(x0, y0, x1, y1):
                mm = int(np.ceil(e)) + 2
                idx = np.nonzero((bb[:, 2] >= x0 - mm) & (bb[:, 0] < x1 + mm) & (bb[:, 3] >= y0 - mm) & (bb[:, 1] < y1 + mm))[0]
                lab = np.zeros((y1 - y0 + 2 * mm, x1 - x0 + 2 * mm), np.int32)
                for j, c in enumerate(idx):
                    cv2.fillPoly(lab, [np.round(polys[c] - (x0 - mm, y0 - mm)).astype(np.int32)], int(j + 1))
                lab = expand_labels(lab, distance=e)[mm:-mm, mm:-mm]
                remap = np.zeros(len(idx) + 1, np.int64); remap[1:] = idx + 1
                return remap[lab]

            cc = np.asarray(cen)[cid]                                    # only blocks holding cells of these units
            keys = np.unique((cc[:, 1] // block).astype(np.int64) * 100000 + (cc[:, 0] // block).astype(np.int64))
            if max_blocks and len(keys) > max_blocks:
                keys = np.sort(np.random.default_rng(0).choice(keys, max_blocks, replace=False))
            S = Cn = None
            for k in keys:
                y0, x0 = int(k // 100000) * block, int(k % 100000) * block
                x1, y1 = min(x0 + block, W), min(y0 + block, H)
                im = reader.read(s, x0 - margin, y0 - margin, T=x1 - x0 + 2 * margin, h=y1 - y0 + 2 * margin)
                p = model.predict_image(im)[:, margin:margin + y1 - y0, margin:margin + x1 - x0].contiguous()
                if S is None:
                    S = torch.zeros(n + 1, p.shape[0], dtype=torch.float64, device=dev)
                    Cn = torch.zeros(n + 1, dtype=torch.float64, device=dev)
                _accumulate(S, Cn, p, torch.from_numpy(labels(x0, y0, x1, y1)).to(dev).reshape(-1))
        pc = (S[1:] / Cn[1:, None].clamp(min=1)).float().cpu().numpy()
        pc[(Cn[1:] == 0).cpu().numpy()] = np.nan
        ok = np.isfinite(pc[cid]).all(1) if max_blocks else np.ones(len(cid), bool)
        out[s] = (sel[ok], pc[cid][ok], B['y'][sel][ok][:, model.keep])
        log(f'  predicted {s}: {int(ok.sum()):,} cells')
    return out
