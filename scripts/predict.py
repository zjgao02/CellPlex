"""Predict the protein levels of every cell of a new H&E image with a trained CELLO model.

Inputs: a checkpoint of scripts/train.py, an H&E image (pyramidal TIFF / SVS readable by OpenSlide, or a uint8
(H, W, 3) .npy) with its resolution, and a CSV of nucleus centroids in level-0 pixels (columns x, y; any other columns,
e.g. a cell id, are copied to the output). As in the transfer experiments, the image is resampled to the model's
training resolution: cells are tiled on a grid of 224 * mpp_model / mpp_image image px, each tile is resized to 224 px
and the centroids are scaled with it. Output: one row per cell with the input columns and one column per protein
(the model's log1p target scale).

  python scripts/predict.py --ckpt runs/multitumor/hoptimus0/fold0/best_model.pt --image slide.svs --mpp 0.25 \
      --cells centroids.csv --out predictions.parquet
"""
import argparse, os, sys, time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cohorts import COHORTS, TILE  # noqa: E402
from cellplex.data import Reader  # noqa: E402
from cellplex.model import load_checkpoint  # noqa: E402


class CellTiles(Dataset):
    """the cells on a grid of R = TILE * mpp_model / mpp_image image px, served at TILE px"""

    def __init__(self, image, mpp_image, mpp_model, cx, cy):
        npy = image.endswith('.npy')
        self.reader = Reader({'reader': 'npy' if npy else 'wsi', 'images': {'img': image}})
        R = TILE * mpp_model / mpp_image
        self.n = int(round(R))
        gx, gy = np.floor(cx / R).astype(np.int64), np.floor(cy / R).astype(np.int64)
        o = np.lexsort((gx, gy))
        key = gy[o] * 10_000_000 + gx[o]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        x0 = np.round(gx[o][starts] * R).astype(np.int64); y0 = np.round(gy[o][starts] * R).astype(np.int64)
        lx = cx[o] - np.repeat(x0, ends - starts); ly = cy[o] - np.repeat(y0, ends - starts)
        self.order = o
        self.lxy = np.clip(np.stack([lx, ly], 1) * (TILE / self.n), 0, TILE - 1e-3).astype(np.float32)
        self.items = list(zip(x0.tolist(), y0.tolist(), starts.tolist(), ends.tolist()))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x0, y0, a, b = self.items[i]
        img = self.reader.read('img', x0, y0, T=self.n)
        if self.n != TILE:
            img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA if self.n > TILE else cv2.INTER_LINEAR)
        return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.), torch.from_numpy(self.lxy[a:b]), (a, b)


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--mpp', type=float, required=True, help='resolution of the image (um/px at level 0)')
    ap.add_argument('--cells', required=True, help='CSV with nucleus centroids x, y in level-0 px')
    ap.add_argument('--out', required=True, help='.parquet or .csv')
    ap.add_argument('--model_mpp', type=float, default=None,
                    help='training resolution of the model (default: that of the checkpoint\'s cohort)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=8)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, proteins = load_checkpoint(a.ckpt, device=dev)
    mpp_model = a.model_mpp or COHORTS[cfg['cohort']]['mpp']
    C = pd.read_csv(a.cells)
    D = CellTiles(a.image, a.mpp, mpp_model, C.x.to_numpy(np.float64), C.y.to_numpy(np.float64))
    print(f'{len(C):,} cells, {len(D):,} tiles of {D.n} px ({a.mpp} -> {mpp_model} um/px), {len(proteins)} proteins, '
          f'{dev}', flush=True)
    pred = np.zeros((len(C), len(proteins)), np.float32)
    dl = DataLoader(D, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=collate, pin_memory=dev.type == 'cuda')
    t0 = time.time()
    with torch.no_grad():
        for imgs, xy, where in dl:
            for (s, e), o in zip(where, model.predict(imgs.to(dev, non_blocking=True), xy)):
                pred[D.order[s:e]] = o.numpy()
    out = pd.concat([C.reset_index(drop=True), pd.DataFrame(pred, columns=proteins)], axis=1)
    out.to_parquet(a.out, index=False) if a.out.endswith('.parquet') else out.to_csv(a.out, index=False)
    print(f'-> {a.out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
