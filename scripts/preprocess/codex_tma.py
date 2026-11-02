"""CRC-TMA and Multi-tumor (Schurch et al. 2020, Cell; TCIA collection CRC_FFPE-CODEX_CellNeighs): CODEX (Keyence
20x, 0.377442 um/px) of two colorectal-cancer TMAs (A, B; 35 patients x 4 cores) and a multi-tumor TMA, with H&E
imaged on the same section after CODEX, on the CODEX pixel grid -> cells and per-cell protein truth.

Raw layout ($CELLPLEX_RAW/{crc_tma,multitumor}/, the TCIA folders as downloaded; data/README.md):
  crc_tma/CRC_TMA_A_HandE/bestFocus/reg###_X01_Y01_Z##.tif         H&E brightfield, (2, 4, H, W) uint16
  crc_tma/CRC_TMA_A_hyperstacks/bestFocus/reg###_X01_Y01_Z##.tif   CODEX, (cycles, 4, H, W) uint16
  crc_tma/CRC_TMA_A_hyperstacks/channelNames.txt, Experiment.json  (and the same for CRC_TMA_B)
  multitumor/Multi-tumor_TMA_HandE/..., multitumor/Multi-tumor_TMA_hyperstacks/...
Samples: crc_tma A_reg001 .. B_reg070, multitumor reg001 .. reg070.

  prep   per core: H&E = the last three brightfield planes, scaled by 65535 / their maximum and floored, then
         uint16 -> uint8 (/ 257, rounded) at native resolution -> he/{s}.npy; CellViT++ input resampled to 0.25 um/px
         (cv_in/{s}.tif + .json) and cellvit_filelist.csv
  (CellViT++: scripts/preprocess/run_cellvit.py process_dataset --filelist cellvit_filelist.csv, outdir seg/)
  quant  nuclei mapped back to native px, region = nucleus + 2 um; per core, the shift that best overlays the nuclei
         on HOECHST1 (FFT cross-correlation, +-40 px) moves the regions into the CODEX frame; truth = mean CODEX
         intensity in the region for every protein of data/panels/{cohort}.csv; QC = cells with Hoechst brighter in
         the nucleus than in the ring, before / after the shift              -> quant/{s}.npz

  python scripts/preprocess/codex_tma.py prep --cohort multitumor [--samples reg001]
  python scripts/preprocess/codex_tma.py quant --cohort multitumor [--samples reg001]
"""
import argparse, glob, json, os, re, sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import PANELS, cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import CV_MPP, cellvit_input, quant_shifted, read_cellvit, write_filelist  # noqa: E402

SETS = {'crc_tma': {'A': 'CRC_TMA_A', 'B': 'CRC_TMA_B'}, 'multitumor': {'': 'Multi-tumor_TMA'}}
HOECHST = ('HOECHST1', 'Hoechst1')


def channel_names(cohort, tma):
    d = f'{raw_dir(cohort)}/{SETS[cohort][tma]}_hyperstacks'
    return [l.strip().split(' - ')[0].strip() for l in open(f'{d}/channelNames.txt') if l.strip()]


def mpp(cohort, tma):
    d = f'{raw_dir(cohort)}/{SETS[cohort][tma]}_hyperstacks'
    return json.load(open(f'{d}/Experiment.json'))['per_pixel_XY_resolution'] / 1000.


def samples(cohort):
    """{sample: (tma, H&E path, CODEX path)}"""
    out = {}
    for tma, name in SETS[cohort].items():
        R = raw_dir(cohort)
        he = {re.match(r'(reg\d+)_X01_Y01_Z\d+\.tif$', os.path.basename(f)).group(1): f
              for f in glob.glob(f'{R}/{name}_HandE/bestFocus/reg*_X01_Y01_Z*.tif')}
        cx = {re.match(r'(reg\d+)_X01_Y01_Z\d+\.tif$', os.path.basename(f)).group(1): f
              for f in glob.glob(f'{R}/{name}_hyperstacks/bestFocus/reg*_X01_Y01_Z*.tif')}
        for k in sorted(set(he) & set(cx)):
            out[f'{tma}_{k}' if tma else k] = (tma, he[k], cx[k])
    return out


def read_he(path):
    """Brightfield planes -> RGB uint8: the last three planes (cycle 2), scaled to 65535 by their joint maximum and
    floored (the release's RGB conversion), then / 257 rounded."""
    a = tifffile.imread(path)
    rgb = a.reshape(-1, *a.shape[-2:])[-3:].astype(np.float64)
    conv = np.floor(rgb * 65535 / rgb.max()).astype(np.uint16)
    return np.ascontiguousarray((np.moveaxis(conv, 0, -1) / 257.).round().astype(np.uint8))


def read_codex(path):
    a = tifffile.imread(path)                                            # (cycles, 4, H, W) uint16
    return a.reshape(-1, *a.shape[-2:])


def prep(a):
    D = cohort_dir(a.cohort)
    for sub in ('he', 'cv_in'):
        os.makedirs(f'{D}/{sub}', exist_ok=True)
    S = samples(a.cohort)
    rows = []
    for s in (a.samples.split(',') if a.samples else list(S)):
        tma, hp, cp = S[s]
        he = read_he(hp)
        with tifffile.TiffFile(cp) as t:
            cs = t.series[0].shape
        assert he.shape[:2] == tuple(cs[-2:]), f'{s}: H&E {he.shape} vs CODEX {cs}'
        np.save(f'{D}/he/{s}.npy', he)
        Ho, Wo = cellvit_input(he, mpp(a.cohort, tma), f'{D}/cv_in/{s}.tif', f'{D}/cv_in/{s}.json', core=s)
        rows.append((f'{D}/cv_in/{s}.tif', CV_MPP, 40))
        print(f'{s}: H&E {he.shape} -> CellViT input {Ho}x{Wo} at {CV_MPP} um/px', flush=True)
    if not a.samples:
        write_filelist(f'{D}/cellvit_filelist.csv', rows)


def quant(a):
    D = cohort_dir(a.cohort)
    os.makedirs(f'{D}/quant', exist_ok=True)
    S = samples(a.cohort)
    panel = list(pd.read_csv(f'{PANELS}/{a.cohort}.csv').marker)
    todo = a.samples.split(',') if a.samples else [s for s in S if os.path.exists(f'{D}/seg/{s}_cells.json')]
    for s in todo:
        tma, _, cp = S[s]
        g = json.load(open(f'{D}/cv_in/{s}.json'))
        polys, cen, typ = read_cellvit(f'{D}/seg/{s}_cells.json', g['sx'], g['sy'])
        cod = read_codex(cp)
        names = channel_names(a.cohort, tma)[:cod.shape[0]]
        hidx = next(names.index(h) for h in HOECHST if h in names)
        Z = quant_shifted(polys, cen, typ, g['H'], g['W'], g['mpp'], cod, hidx, {p: names.index(p) for p in panel})
        np.savez_compressed(f'{D}/quant/{s}.npz', **Z)
        print(f'{s}: {len(cen):,} nuclei, shift CODEX = H&E + ({Z["shift"][0]}, {Z["shift"][1]}) px, Hoechst nucleus > '
              f'ring {Z["hoechst_nuc_gt_ring_unshifted"]:.1%} -> {Z["hoechst_nuc_gt_ring"]:.1%}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prep', 'quant'])
    ap.add_argument('--cohort', required=True, choices=list(SETS))
    ap.add_argument('--samples', default='', help='comma-separated (default: all)')
    a = ap.parse_args()
    {'prep': prep, 'quant': quant}[a.step](a)


if __name__ == '__main__':
    main()
