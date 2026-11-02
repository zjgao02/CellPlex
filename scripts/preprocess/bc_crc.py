"""BC/CRC (Matusiak et al. 2024, Cancer Discovery): CODEX of breast (BC) and colorectal (CRC) cancer TMA regions,
0.5072 um/px, with H&E montages on the CODEX pixel grid (3468 x 4620 px per region) -> cells and per-cell protein
truth, the protocol of scripts/preprocess/codex_tma.py.

The images used here were obtained from the authors: H&E montages and 8-bit CODEX montages (80 channels for BC, 76
for CRC) on one pixel grid per region. The public release of the study holds 16-bit CODEX stacks only, from another
export and without H&E (data/README.md).

Raw layout ($CELLPLEX_RAW/bc_crc/):
  HE/{BC,CRC}_reg{NNN}.HE.tif                           H&E, (H, W, 3) uint8
  IF_uint8/{BC,CRC}_reg{NNN}.IF.tif                     CODEX, (C, H, W) uint8
  channel tables: data/panels/bc_crc_channels_{BC,CRC}.csv (Marker, Index, Included, MarkerUnified); the 36 channels
  marked Included in both panels (not Hoechst, DRAQ5) are the panel, in the CRC panel's order

  prep   per region ({BC,CRC}_reg{NNN}, regions with both an H&E and a CODEX): he/{region}.npy and the CellViT++
         input at 0.25 um/px (cv_in/) + cellvit_filelist.csv
  (CellViT++: scripts/preprocess/run_cellvit.py process_dataset --filelist cellvit_filelist.csv, outdir seg/)
  quant  region = nucleus + 2 um, shift on Hoechst1 (~0 here), truth = mean 8-bit CODEX intensity -> quant/{region}.npz

  python scripts/preprocess/bc_crc.py prep [--regions BC_reg001]
  python scripts/preprocess/bc_crc.py quant [--regions BC_reg001]
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import PANELS, cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import CV_MPP, cellvit_input, quant_shifted, read_cellvit, write_filelist  # noqa: E402

SRC, D = raw_dir('bc_crc'), cohort_dir('bc_crc')
MPP = 0.5072


def files(region):
    return f'{SRC}/HE/{region}.HE.tif', f'{SRC}/IF_uint8/{region}.IF.tif'


def regions():
    """Regions with both an H&E and a CODEX (46)."""
    he = {os.path.basename(f)[:-7] for f in glob.glob(f'{SRC}/HE/*.HE.tif')}
    return sorted(he & {os.path.basename(f)[:-7] for f in glob.glob(f'{SRC}/IF_uint8/*.IF.tif')})


def panel(region):
    """{protein: channel index} for the region's panel, in the CRC panel's order; Hoechst1 index."""
    ch = pd.read_csv(f'{PANELS}/bc_crc_channels_{region.split("_")[0]}.csv')
    order = pd.read_csv(f'{PANELS}/bc_crc_channels_CRC.csv')
    keep = [m for m in order[order.Included == 1].MarkerUnified if m not in ('Hoechst', 'DRAQ5')]
    idx = dict(zip(ch[ch.Included == 1].MarkerUnified, ch[ch.Included == 1].Index))
    return {p: int(idx[p]) for p in keep}, int(ch.index[ch.Marker == 'Hoechst1'][0])


def prep(a):
    for sub in ('he', 'cv_in'):
        os.makedirs(f'{D}/{sub}', exist_ok=True)
    rows = []
    for r in (a.regions.split(',') if a.regions else regions()):
        hp, cp = files(r)
        he = np.ascontiguousarray(tifffile.imread(hp)[..., :3].astype(np.uint8))
        with tifffile.TiffFile(cp) as t:
            cs = t.series[0].shape
        assert he.shape[:2] == tuple(cs[1:]), f'{r}: H&E {he.shape} vs CODEX {cs}'
        np.save(f'{D}/he/{r}.npy', he)
        Ho, Wo = cellvit_input(he, MPP, f'{D}/cv_in/{r}.tif', f'{D}/cv_in/{r}.json', region=r)
        rows.append((f'{D}/cv_in/{r}.tif', CV_MPP, 40))
        print(f'{r}: H&E {he.shape} -> CellViT input {Ho}x{Wo} at {CV_MPP} um/px', flush=True)
    if not a.regions:
        write_filelist(f'{D}/cellvit_filelist.csv', rows)


def quant(a):
    os.makedirs(f'{D}/quant', exist_ok=True)
    todo = a.regions.split(',') if a.regions else [r for r in regions() if os.path.exists(f'{D}/seg/{r}_cells.json')]
    for r in todo:
        g = json.load(open(f'{D}/cv_in/{r}.json'))
        polys, cen, typ = read_cellvit(f'{D}/seg/{r}_cells.json', g['sx'], g['sy'])
        pidx, hidx = panel(r)
        Z = quant_shifted(polys, cen, typ, g['H'], g['W'], MPP, tifffile.imread(files(r)[1]), hidx, pidx)
        np.savez_compressed(f'{D}/quant/{r}.npz', **Z, tissue=r.split('_')[0])
        print(f'{r}: {len(cen):,} nuclei, shift CODEX = H&E + ({Z["shift"][0]}, {Z["shift"][1]}) px, Hoechst nucleus > '
              f'ring {Z["hoechst_nuc_gt_ring_unshifted"]:.1%} -> {Z["hoechst_nuc_gt_ring"]:.1%}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prep', 'quant'])
    ap.add_argument('--regions', default='')
    a = ap.parse_args()
    {'prep': prep, 'quant': quant}[a.step](a)


if __name__ == '__main__':
    main()
