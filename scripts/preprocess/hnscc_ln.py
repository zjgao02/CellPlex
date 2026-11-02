"""HNSCC-LN (Haist et al. 2026, Cancer Cell; BioImage Archive S-BIAD1638, discovery cohort): CODEX of 3 TMAs (521-523,
2 slides each, 390 cores of primary tumours and lymph nodes) with H&E on the same section, Keyence 20x
(0.377442 um/px), one 1440 x 1920 image per core on the CODEX grid -> cells and per-cell protein truth.

Raw layout ($CELLPLEX_RAW/hnscc_ln/; data/urls/hnscc_ln.txt):
  raw_TIFF/{tma}_S{k}reg###.tif               CODEX, (30 cycles, 4, H, W) uint16
  raw_TIFF/{tma}_S{k}_channelnames.txt        120 channel names, cycle-major
  HandE/{tma}_S{k}reg###_HandE.tif            H&E, (3, H, W) uint16 holding 0..255
  metadata/HNSCCdiscovery_clinmetadata.csv    patient (CaseID) and nodal status of every core (Mendeley Data)

  prep   H&E -> uint8; black padding connected to the image border (from the authors' H&E -> CODEX warp) -> white;
         he/{core}.npy and the CellViT++ input at 0.25 um/px (cv_in/) + cellvit_filelist.csv
  (CellViT++: scripts/preprocess/run_cellvit.py process_dataset --filelist cellvit_filelist.csv, outdir seg/)
  quant  as scripts/preprocess/codex_tma.py: region = nucleus + 2 um, per-core shift on Hoechst1, truth = mean CODEX
         intensity for every protein channel (not Hoechst, blank, empty, DRAQ5)  -> quant/{core}.npz

  python scripts/preprocess/hnscc_ln.py prep [--cores 521_S1reg001]
  python scripts/preprocess/hnscc_ln.py quant [--cores 521_S1reg001]
"""
import argparse, glob, json, os, re, sys

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import CV_MPP, cellvit_input, quant_shifted, read_cellvit, write_filelist  # noqa: E402

SRC, D = raw_dir('hnscc_ln'), cohort_dir('hnscc_ln')
MPP = 0.377442
NOT_PROTEIN = re.compile(r'(?i)^(hoe?chst|blank|empty|draq5)')


def cores():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{SRC}/raw_TIFF/*reg*.tif'))


def channels(core):
    """(names of the 120 channels in hyperstack order: cycle-major, 4 per cycle), {protein: index}, Hoechst1 index."""
    names = [l.strip() for l in open(f'{SRC}/raw_TIFF/{core.split("reg")[0]}_channelnames.txt') if l.strip()]
    prot = {n: i for i, n in enumerate(names) if not NOT_PROTEIN.match(n)}
    return names, prot, names.index('Hoechst1')


def read_codex(core):
    a = tifffile.imread(f'{SRC}/raw_TIFF/{core}.tif')                    # (30 cycles, 4, H, W) uint16
    return a.reshape(-1, *a.shape[-2:])


def read_he(core):
    he = tifffile.imread(f'{SRC}/HandE/{core}_HandE.tif')                # (3, H, W) uint16 holding 0..255
    he = np.moveaxis(he, 0, -1) if he.shape[0] == 3 else he
    he = np.ascontiguousarray(np.clip(he, 0, 255).astype(np.uint8)[..., :3])
    black = (he.max(2) <= 3).astype(np.uint8)                          # padding from the authors' H&E -> CODEX warp
    n, lab, st, _ = cv2.connectedComponentsWithStats(black, connectivity=4)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    he[np.isin(lab, edge[edge > 0])] = 255                              # -> white background, as outside a WSI
    return he


def prep(a):
    for sub in ('he', 'cv_in'):
        os.makedirs(f'{D}/{sub}', exist_ok=True)
    rows = []
    for c in (a.cores.split(',') if a.cores else cores()):
        he = read_he(c)
        with tifffile.TiffFile(f'{SRC}/raw_TIFF/{c}.tif') as t:
            cs = t.series[0].shape
        assert he.shape[:2] == tuple(cs[-2:]), f'{c}: H&E {he.shape} vs CODEX {cs}'
        np.save(f'{D}/he/{c}.npy', he)
        Ho, Wo = cellvit_input(he, MPP, f'{D}/cv_in/{c}.tif', f'{D}/cv_in/{c}.json', core=c)
        rows.append((f'{D}/cv_in/{c}.tif', CV_MPP, 40))
        print(f'{c}: H&E {he.shape} -> CellViT input {Ho}x{Wo}', flush=True)
    if not a.cores:
        write_filelist(f'{D}/cellvit_filelist.csv', rows)


def quant(a):
    os.makedirs(f'{D}/quant', exist_ok=True)
    todo = a.cores.split(',') if a.cores else [c for c in cores() if os.path.exists(f'{D}/seg/{c}_cells.json')]
    for c in todo:
        g = json.load(open(f'{D}/cv_in/{c}.json'))
        polys, cen, typ = read_cellvit(f'{D}/seg/{c}_cells.json', g['sx'], g['sy'])
        if not len(cen):
            print(f'{c}: no nuclei', flush=True)
            continue
        _, pidx, hidx = channels(c)
        Z = quant_shifted(polys, cen, typ, g['H'], g['W'], MPP, read_codex(c), hidx, pidx)
        np.savez_compressed(f'{D}/quant/{c}.npz', **Z)
        print(f'{c}: {len(cen):,} nuclei, shift CODEX = H&E + ({Z["shift"][0]}, {Z["shift"][1]}) px, Hoechst nucleus > '
              f'ring {Z["hoechst_nuc_gt_ring_unshifted"]:.1%} -> {Z["hoechst_nuc_gt_ring"]:.1%}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prep', 'quant'])
    ap.add_argument('--cores', default='')
    a = ap.parse_args()
    {'prep': prep, 'quant': quant}[a.step](a)


if __name__ == '__main__':
    main()
