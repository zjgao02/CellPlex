"""Bone marrow (Bandyopadhyay et al. 2024, Cell; Figshare+ collection 10.25452/figshare.plus.c.7174914): PhenoCycler
Fusion CODEX (54 channels, 0.5069 um/px) and H&E stained on the same section after CODEX (0.2485 um/px), 12 sections
(10 normal, 1 AML, 1 negative staging marrow) -> cells and per-cell protein truth.

The truth of CellPlex was read from CODEX images registered to the H&E pixel grid (54 channels, uint8, same pixel =
same location), which were provided with the data and are not part of the public release. Without them, `register`
registers the public CODEX to the H&E with the method used for the glioma cohort (cellplex/preprocess/register.py)
and `quant --source registered` samples the CODEX through that registration; this path has not been compared
against the registered images used for CellPlex.

Raw layout ($CELLPLEX_RAW/bone_marrow/; data/urls/bone_marrow.txt):
  HE/{case}_HE.qptiff                  H&E (Figshare+ 'Normal Bone Marrow H&E Images', 'AML and NSM H&E Images')
  IF/{case}_Scan1.qptiff               CODEX (Figshare+ '... CODEX Images'), channels in MarkerList.txt order
  aligned/{case}.IF.ome.tiff           optional: CODEX on the H&E grid (the files used for CellPlex)
  channel table: data/panels/bone_marrow_channels.csv (panel A for AML / NSM cases, B for the normal cases)

  he        H&E .qptiff -> standard pyramidal TIFF on the same pixel grid (he_tif/{case}.tif; JPEG q95 tiles; the
            qptiff holds several Subfile-Type-0 images, which CellViT++ rejects) + cellvit_filelist.csv
  (CellViT++ at the native 0.2485 um/px: scripts/preprocess/run_cellvit.py process_dataset, outdir seg/)
  register  optional, see above                                                    -> reg/{case}.npz, _qc.jpg
  quant     region = nucleus + 2 um, value = mean CODEX intensity in the region per channel, in 4096 px blocks;
            QC = cells whose DAPI is brighter in the nucleus than in the ring         -> quant/{case}.npz

  python scripts/preprocess/bone_marrow.py he --case H35
  python scripts/preprocess/bone_marrow.py quant --case H35 --workers 16 [--source registered]
"""
import argparse, glob, os, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import tifffile
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import PANELS, cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import EXPAND_UM, read_cellvit, write_filelist  # noqa: E402
from cellplex.preprocess.register import register, sample_on_grid  # noqa: E402

SRC, D = raw_dir('bone_marrow'), cohort_dir('bone_marrow')
MPP, MPP_CX = 0.2485, 0.5069
BLOCK, TS = 4096, 256
G = {}


def cases():
    return sorted(os.path.basename(f)[:-10] for f in glob.glob(f'{SRC}/HE/*_HE.qptiff'))


def channel_names(case):
    ch = pd.read_csv(f'{PANELS}/bone_marrow_channels.csv')
    ch = ch[ch.panel.str.startswith('B' if case.startswith('H') else 'A')].sort_values('channel_index')
    return list(ch.unified_marker)


def he_path(case):
    return f'{SRC}/HE/{case}_HE.qptiff'


def if_path(case, source):
    return f'{SRC}/aligned/{case}.IF.ome.tiff' if source == 'aligned' else glob.glob(f'{SRC}/IF/{case}_Scan*.qptiff')[0]


def tiles(z):
    """Row-major 256 px tiles of one pyramid level (white outside the image)."""
    H, W = z.shape[:2]
    for y in range(0, H, TS):
        strip = np.asarray(z[y:min(y + TS, H)])
        for x in range(0, W, TS):
            t = np.full((TS, TS, 3), 255, np.uint8)
            c = strip[:TS, x:x + TS]
            t[:c.shape[0], :c.shape[1]] = c
            yield t


def convert_he(case):
    """Level 0 is copied pixel for pixel, so cells, H&E and the registered CODEX keep sharing one grid; the qptiff's own
    2x pyramid levels are reused."""
    t0 = time.time()
    with tifffile.TiffFile(he_path(case)) as t:
        nlev = len(t.series[0].levels)
    Z = [zarr.open(tifffile.imread(he_path(case), aszarr=True, level=k), mode='r') for k in range(nlev)]
    H, W = Z[0].shape[:2]
    os.makedirs(f'{D}/he_tif', exist_ok=True)
    out, tmp = f'{D}/he_tif/{case}.tif', f'{D}/he_tif/{case}.tmp.tif'
    opts = dict(tile=(TS, TS), compression='jpeg', compressionargs={'level': 95}, photometric='rgb',
                resolutionunit='CENTIMETER', dtype=np.uint8)
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw:
        tw.write(tiles(Z[0]), shape=(H, W, 3), resolution=(1e4 / MPP, 1e4 / MPP), subifds=0, **opts)
        for k in range(1, nlev):
            h, w = Z[k].shape[:2]
            f = W / w
            tw.write(tiles(Z[k]), shape=(h, w, 3), subfiletype=1, resolution=(1e4 / (MPP * f), 1e4 / (MPP * f)), **opts)
    os.replace(tmp, out)
    print(f'{case}: {W}x{H} -> {out} ({nlev} levels, {os.path.getsize(out) / 1e9:.1f} GB, {time.time() - t0:.0f}s)', flush=True)


def work(task):
    from skimage.segmentation import expand_labels
    by, bx = task
    polys, bb, H, W, C, e = G['polys'], G['bb'], G['H'], G['W'], G['C'], G['e']
    m = int(np.ceil(e)) + 2
    x0, y0 = bx * BLOCK, by * BLOCK
    x1, y1 = min(x0 + BLOCK, W), min(y0 + BLOCK, H)
    idx = np.nonzero((bb[:, 2] >= x0 - m) & (bb[:, 0] < x1 + m) & (bb[:, 3] >= y0 - m) & (bb[:, 1] < y1 + m))[0]
    if not len(idx):
        return None
    lab = np.zeros((y1 - y0 + 2 * m, x1 - x0 + 2 * m), np.int32)
    for j, c in enumerate(idx):
        cv2.fillPoly(lab, [np.round(polys[c] - (x0 - m, y0 - m)).astype(np.int32)], int(j + 1))
    reg = expand_labels(lab, distance=e)
    lab, reg = lab[m:-m, m:-m], reg[m:-m, m:-m]
    if 'z' not in G:
        G['z'] = zarr.open(tifffile.imread(G['if_path'], aszarr=True, level=0), mode='r')
    a = np.asarray(G['z'][:, y0:y1, x0:x1]).reshape(C, -1)             # converted one channel at a time
    n = len(idx)
    l, ln = reg.ravel(), lab.ravel()
    area = np.bincount(l, minlength=n + 1)[1:]
    area_n = np.bincount(ln, minlength=n + 1)[1:]
    S = np.stack([np.bincount(l, weights=a[k].astype(np.float32), minlength=n + 1)[1:] for k in range(C)], 1)
    Sd = np.bincount(ln, weights=a[G['dapi']].astype(np.float32), minlength=n + 1)[1:]
    keep = area > 0
    return idx[keep], area[keep], area_n[keep], S[keep], Sd[keep]


def quant(case, workers, source):
    t0 = time.time()
    os.makedirs(f'{D}/quant', exist_ok=True)
    names = channel_names(case)
    polys, cen, typ = read_cellvit(f'{D}/seg/{case}_cells.json')
    n = len(polys)
    if source == 'registered':                                         # CODEX sampled through reg/{case}.npz
        reg_ = np.load(f'{D}/reg/{case}.npz')
        z = zarr.open(tifffile.imread(if_path(case, source), aszarr=True, level=0), mode='r')
        H, W = [int(v) for v in reg_['he_shape']]
        mean, AREA, AN, BAD, qc = sample_on_grid(reg_, z, names, names, polys, H, W, MPP,
                                                 log=lambda m: print(f'  {case}: {m}', flush=True))
    else:                                                              # CODEX already on the H&E grid
        with tifffile.TiffFile(if_path(case, source)) as t:
            C, H, W = t.series[0].shape
        bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
        G.update(polys=polys, bb=bb, H=H, W=W, C=C, e=EXPAND_UM / MPP, if_path=if_path(case, source), dapi=names.index('DAPI'))
        print(f'{case}: {n:,} nuclei, CODEX {C} x {H} x {W} ({time.time() - t0:.0f}s)', flush=True)
        AREA = np.zeros(n, np.int64); AN = np.zeros(n, np.int64); SUM = np.zeros((n, C)); SD = np.zeros(n)
        tasks = [(by, bx) for by in range((H + BLOCK - 1) // BLOCK) for bx in range((W + BLOCK - 1) // BLOCK)]
        with Pool(workers) as pool:
            for k, r in enumerate(pool.imap_unordered(work, tasks)):
                if r is not None:
                    idx, ar, an, S, sd = r
                    AREA[idx] += ar; AN[idx] += an; SUM[idx] += S; SD[idx] += sd
                if k % 50 == 0:
                    print(f'  block {k}/{len(tasks)} ({time.time() - t0:.0f}s)', flush=True)
        mean = (SUM / np.maximum(AREA, 1)[:, None]).astype(np.float32)
        mean[AREA == 0] = np.nan
        dn = SD / np.maximum(AN, 1)
        ring = (SUM[:, names.index('DAPI')] - SD) / np.maximum(AREA - AN, 1)
        q = (AN > 0) & (AREA - AN > 0)
        qc = float(np.mean(dn[q] > ring[q]))
    np.savez_compressed(f'{D}/quant/{case}.npz', cx=cen[:, 0].astype(np.float32), cy=cen[:, 1].astype(np.float32),
                        cls=np.array(typ), area=AREA.astype(np.int32), area_nuc=AN.astype(np.int32), mean=mean,
                        channels=np.array(names), panel='B' if case.startswith('H') else 'A', dapi_nuc_gt_ring=qc,
                        mpp=MPP, expand_um=EXPAND_UM, source=source)
    print(f'{case}: done, DAPI nucleus > ring {qc:.1%}, cells with empty region {(AREA == 0).sum():,} '
          f'({time.time() - t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['he', 'register', 'quant'])
    ap.add_argument('--case', default='', help='comma-separated (default: all)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--source', default='aligned', choices=['aligned', 'registered'],
                    help='quant: CODEX on the H&E grid (aligned/) or the public CODEX through reg/{case}.npz')
    a = ap.parse_args()
    todo = a.case.split(',') if a.case else cases()
    if a.step == 'he':
        for c in todo:
            convert_he(c)
        if not a.case:
            write_filelist(f'{D}/cellvit_filelist.csv', [(f'{D}/he_tif/{c}.tif', MPP, 40) for c in todo])
    elif a.step == 'register':
        os.makedirs(f'{D}/reg', exist_ok=True)
        for c in todo:
            register(f'{D}/he_tif/{c}.tif', if_path(c, 'registered'), channel_names(c).index('DAPI'), MPP, MPP_CX,
                     f'{D}/reg/{c}.npz', f'{D}/reg/{c}_qc.jpg', log=lambda m: print(f'{c}: {m}', flush=True))
    else:
        for c in todo:
            quant(c, a.workers, a.source)


if __name__ == '__main__':
    main()
