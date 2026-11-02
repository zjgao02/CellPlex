"""Glioma (Hoefflin et al. 2026, Cancer Cell; BioImage Archive S-BIAD2840): 37 whole sections (IDH-mutant
astrocytomas and oligodendrogliomas, GBM; 30 patients), PhenoCycler Fusion CODEX (0.5067 um/px) + H&E stained on the
same section after CODEX (0.2485 um/px, not on the CODEX grid) -> cells and per-cell protein truth.

Raw layout ($CELLPLEX_RAW/glioma/; data/urls/glioma.txt):
  scans_IDHm/IDHm_exp{1..6}/{id}.ome.tif, {id}_HE.ome.tif     CODEX (69, H, W) uint16 pyramid; H&E RGB pyramid
  scans_GBM/GBM_exp{1,2,4,8}/{id}.ome.tif, {id}_HE.ome.tif
  TableS1_SampleMetadata_v13.xlsx                            sample -> patient, tumour type (Zenodo 21335411)

  prep      H&E level 0 -> pyramidal JPEG TIFF at 0.2485 um/px (he_tif/{s}.tif: CellViT's x40 without rescaling and
            the models' WSI input) + cellvit_filelist.csv
  register  CODEX -> H&E: affine + local displacement field (cellplex/preprocess/register.py), from the source H&E
            pyramid (he_tif/ when the source cannot be decoded)                  -> reg/{s}.npz, _qc.jpg
  (CellViT++ with CELLVIT_ALL_PATCHES=1: Otsu tissue detection misses most of these pale sections;
   scripts/preprocess/run_cellvit.py process_dataset --filelist cellvit_filelist.csv, outdir seg/)
  quant     per 4096 px H&E block: nuclei rasterised, region = nucleus + 2 um, every region pixel mapped to the CODEX
            frame and sampled bilinearly; truth = mean CODEX intensity for the 68 markers of data/panels/glioma.csv;
            cells with a region pixel outside the imaged CODEX field are marked `outside` (dropped);
            QC = cells whose DAPI is brighter in the nucleus than in the ring       -> quant/{s}.npz

  python scripts/preprocess/glioma.py prep --samples GBM27
  python scripts/preprocess/glioma.py register --samples GBM27 [--coarse intensity]
  python scripts/preprocess/glioma.py quant --samples GBM27
"""
import argparse, glob, os, re, sys

import numpy as np
import pandas as pd
import tifffile
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import PANELS, cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import EXPAND_UM, read_cellvit, write_filelist, write_pyramid  # noqa: E402
from cellplex.preprocess.register import register, sample_on_grid  # noqa: E402

SRC, D = raw_dir('glioma'), cohort_dir('glioma')
MPP_HE, MPP_CX = 0.2485, 0.5067


def files():
    """{sample: (CODEX path, H&E path)}"""
    return {os.path.basename(f)[:-8]: (f, f[:-8] + '_HE.ome.tif') for f in sorted(glob.glob(f'{SRC}/scans_*/*/*.ome.tif'))
            if not f.endswith('_HE.ome.tif')}


def channel_names(path):
    with tifffile.TiffFile(path) as t:
        return re.findall(r'<Channel [^>]*Name="([^"]*)"', t.ome_metadata or '')


def sample_table():
    """SampleID, PatientID, GliomaType of the CODEX sections (IDH-mutant cohort + the GBMs of the previous cohort)."""
    x = pd.ExcelFile(f'{SRC}/TableS1_SampleMetadata_v13.xlsx')
    a = x.parse('This_cohort')
    b = x.parse('Previous_cohort')
    b = b[b.CODEX == 1].assign(GliomaType='GBM')
    return pd.concat([a[a.CODEX == 1][['SampleID', 'PatientID', 'GliomaType']], b[['SampleID', 'PatientID', 'GliomaType']]])


def prep(a):
    os.makedirs(f'{D}/he_tif', exist_ok=True)
    F = files()
    for s in (a.samples.split(',') if a.samples else list(F)):
        he = tifffile.imread(F[s][1], series=0, level=0)                  # (H, W, 3) uint8
        write_pyramid(f'{D}/he_tif/{s}.tif', np.ascontiguousarray(he[..., :3]), MPP_HE)
        print(f'{s}: H&E {he.shape} -> he_tif', flush=True)
    if not a.samples:
        write_filelist(f'{D}/cellvit_filelist.csv', [(f'{D}/he_tif/{s}.tif', MPP_HE, 40) for s in F])


def reg(a):
    os.makedirs(f'{D}/reg', exist_ok=True)
    F = files()
    for s in (a.samples.split(',') if a.samples else list(F)):
        cx_path, he_path = F[s]
        conv = f'{D}/he_tif/{s}.tif'                       # fallback when the source pyramid has corrupt J2K tiles
        dapi = next(i for i, n in enumerate(channel_names(cx_path)) if re.match(r'(?i)dapi|hoechst', n))
        for k, p in enumerate((he_path, conv)):
            try:
                register(p, cx_path, dapi, MPP_HE, MPP_CX, f'{D}/reg/{s}.npz', f'{D}/reg/{s}_qc.jpg', a.coarse,
                         log=lambda m: print(f'{s}: {m}', flush=True))
                break
            except Exception as e:
                if k == 1 or not os.path.exists(conv):
                    raise
                print(f'{s}: source H&E pyramid unreadable ({type(e).__name__}), using he_tif/', flush=True)


def quant(a):
    os.makedirs(f'{D}/quant', exist_ok=True)
    F = files()
    prots = list(pd.read_csv(f'{PANELS}/glioma.csv').marker)
    todo = a.samples.split(',') if a.samples else [s for s in F if os.path.exists(f'{D}/seg/{s}_cells.json')]
    for s in todo:
        cx_path = F[s][0]
        reg_ = np.load(f'{D}/reg/{s}.npz')
        polys, cen, typ = read_cellvit(f'{D}/seg/{s}_cells.json')
        z = zarr.open(tifffile.imread(cx_path, aszarr=True, level=0), mode='r')
        H, W = [int(v) for v in reg_['he_shape']]
        mean, AREA, AN, BAD, qc = sample_on_grid(reg_, z, channel_names(cx_path), prots, polys, H, W, MPP_HE,
                                                 log=lambda m: print(f'  {s}: {m}', flush=True))
        np.savez_compressed(f'{D}/quant/{s}.npz', keys_all=np.array(prots), mean_all=mean, cx=cen[:, 0].astype(np.float32),
                            cy=cen[:, 1].astype(np.float32), cls=np.array(typ), area=AREA.astype(np.int32),
                            area_nuc=AN.astype(np.int32), outside=BAD, mpp=MPP_HE, expand_um=EXPAND_UM, dapi_nuc_gt_ring=qc)
        print(f'{s}: {len(polys):,} nuclei, {BAD.sum():,} outside the CODEX field, {len(prots)} markers, '
              f'DAPI nucleus > ring {qc:.1%}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prep', 'register', 'quant'])
    ap.add_argument('--samples', default='')
    ap.add_argument('--coarse', default='mask', choices=['mask', 'intensity'],
                    help='register: intensity when the tissue fills the CODEX field')
    a = ap.parse_args()
    {'prep': prep, 'register': reg, 'quant': quant}[a.step](a)


if __name__ == '__main__':
    main()
