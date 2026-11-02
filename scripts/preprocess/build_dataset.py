"""Build one processed cohort (the format read by cellplex/data.py) from the outputs of its preprocessing script.

  meta.json            proteins, mpp, reader, image path template, unit, QC and exclusions
  cells/{sample}.npz   cx, cy (float32, native px, sorted by tile), y (n, P) float32 log1p mean intensity in the cell
                       region (nucleus + 2 um), cell_id (index in the source cell table: quant/{sample}.npz)
  tiles.csv            sample, x0, y0, c0, c1 (cell range in cells/{sample}.npz), unit (split unit)
  folds.csv            unit, fold: data/splits/{cohort}.csv

Tiles: non-overlapping 224 px grid from (0, 0) at native resolution; a cell belongs to the tile holding its centroid;
tiles without cells are dropped; cells without a finite truth are dropped. Samples whose nuclear stain is brighter in
the nucleus than in its 2 um ring for fewer than 80% of cells (H&E <-> multiplex misaligned) are excluded.

  python scripts/preprocess/build_dataset.py --cohort multitumor
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.cohorts import COHORTS, TILE  # noqa: E402
from cellplex.paths import PANELS, SPLITS, cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.common import MIN_QC  # noqa: E402

TRUTH = 'log1p(mean intensity in CellViT++ nucleus + 2 um)'
EXCLUSION = f'nuclear stain brighter in the nucleus than in its 2 um ring for < {MIN_QC:.0%} of cells'


def image(cohort):
    d = cohort_dir(cohort)
    return {'orion_crc': f'{d}/wsi/{{sample}}.tif', 'bone_marrow': f'{d}/he_tif/{{sample}}.tif',
            'glioma': f'{d}/he_tif/{{sample}}.tif'}.get(cohort, f'{d}/he/{{sample}}.npy')


def write(cohort, samples, proteins, units, extra):
    """samples: {sample: (cx, cy, y, cell_id)}; units: {sample: unit}."""
    d = cohort_dir(cohort)
    panel = list(pd.read_csv(f'{PANELS}/{cohort}.csv').marker)
    assert list(proteins) == panel, f'{cohort}: proteins differ from data/panels/{cohort}.csv'
    F = pd.read_csv(f'{SPLITS}/{cohort}.csv', dtype={'unit': str})
    missing = set(F.unit) - set(units.values())
    if missing:
        print(f'WARNING {cohort}: {len(missing)} split units without a sample, e.g. {sorted(missing)[:5]}', flush=True)
    os.makedirs(f'{d}/cells', exist_ok=True)
    rows = []
    for s, (cx, cy, y, cid) in samples.items():
        if units[s] not in set(F.unit):
            continue
        ok = np.isfinite(y).all(1) & np.isfinite(cx) & np.isfinite(cy) & (cx >= 0) & (cy >= 0)
        cx, cy, y, cid = cx[ok], cy[ok], y[ok], cid[ok]
        tx, ty = (cx // TILE).astype(np.int64), (cy // TILE).astype(np.int64)
        key = ty * 100000 + tx
        o = np.argsort(key, kind='stable')
        cx, cy, y, cid, key = cx[o], cy[o], y[o], cid[o], key[o]
        np.savez(f'{d}/cells/{s}.npz', cx=cx.astype(np.float32), cy=cy.astype(np.float32), y=y.astype(np.float32),
                 cell_id=cid.astype(np.int64))
        uk, c0 = np.unique(key, return_index=True)
        c1 = np.r_[c0[1:], len(key)]
        for k, a, b in zip(uk, c0, c1):
            rows.append((s, int(k % 100000) * TILE, int(k // 100000) * TILE, int(a), int(b), units[s]))
        print(f'{s}: {len(cx):,} cells, {len(uk):,} tiles', flush=True)
    T = pd.DataFrame(rows, columns=['sample', 'x0', 'y0', 'c0', 'c1', 'unit'])
    T.to_csv(f'{d}/tiles.csv', index=False)
    F.to_csv(f'{d}/folds.csv', index=False)
    c = COHORTS[cohort]
    json.dump({'dataset': cohort, 'name': c['name'], 'proteins': list(proteins), 'tile': TILE, 'mpp': c['mpp'],
               'reader': c['reader'], 'image': image(cohort), 'unit': c['unit'], 'folds': c['folds'], **extra,
               'truth': TRUTH, 'n_cells': int(T.c1.sub(T.c0).sum()), 'n_tiles': len(T),
               'n_samples': int(T['sample'].nunique())}, open(f'{d}/meta.json', 'w'), indent=1)
    return T


def from_quant(cohort, qc_key='hoechst_nuc_gt_ring', unit=lambda s: s):
    """Array / slide cohorts with quant/{s}.npz (keys_all, mean_all, cx, cy): truth = log1p(mean_all)."""
    d = cohort_dir(cohort)
    samples, proteins, qc, dropped = {}, None, {}, {}
    for f in sorted(os.listdir(f'{d}/quant')):
        s = f[:-4]
        Z = np.load(f'{d}/quant/{f}', allow_pickle=True)
        q = float(Z[qc_key])
        u = unit(s)
        if u is None or not q >= MIN_QC:                                   # no metadata, or still misaligned
            dropped[s] = round(q, 3) if u is not None else 'no metadata'
            continue
        keys = [str(k) for k in Z['keys_all']]
        proteins = proteins or keys
        assert keys == proteins
        samples[s] = (Z['cx'].astype(np.float64), Z['cy'].astype(np.float64), np.log1p(Z['mean_all'].astype(np.float64)),
                      np.arange(len(Z['cx'])))
        qc[s] = round(q, 3)
    return samples, proteins, {s: unit(s) for s in samples}, {'qc_nucleus_gt_ring': qc, 'excluded': dropped,
                                                              'exclusion': EXCLUSION}


def orion_crc():
    """Cells of seg/{slide}.parquet in the order of the 512 px tiles of wsi/{slide}.json (stable), truth = log1p of the
    nucleus + 2 um mean of the 16 proteins (quant/{slide}.npz)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import orion_crc as O
    d = cohort_dir('orion_crc')
    samples, units = {}, {}
    for s in sorted(f[:-4] for f in os.listdir(f'{d}/quant') if f.endswith('.npz')):
        Q = np.load(f'{d}/quant/{s}.npz', allow_pickle=True)
        k = list(np.round(Q['expand_um'], 3)).index(2.0)
        markers = [str(m) for m in Q['markers']]
        y = np.log1p(Q['mean'][:, k, [markers.index(m) for m in O.PROTEINS]].astype(np.float64))
        W, H = (json.load(open(f'{d}/wsi/{s}.json'))[v] for v in ('W', 'H'))
        nx, ny = W // O.TILE, H // O.TILE
        cx, cy = Q['cx'], Q['cy']
        key = np.clip((cy // O.TILE).astype(np.int64), 0, ny - 1) * nx + np.clip((cx // O.TILE).astype(np.int64), 0, nx - 1)
        o = np.argsort(key, kind='stable')
        samples[s] = (np.asarray(cx[o], np.float64), np.asarray(cy[o], np.float64), np.asarray(y[o].astype(np.float32), np.float64), o)
        units[s] = s.split('_')[0]                                          # CRC33_Scan1 / _Scan2: one patient
    write('orion_crc', samples, O.PROTEINS, units, {'panel': '16 Orion proteins', 'source': 'Lin et al. 2023'})


def crc_tma():
    S, P, U, X = from_quant('crc_tma')
    write('crc_tma', S, P, U, {'panel': f'all {len(P)} protein channels', 'source': 'TCIA CRC_FFPE-CODEX_CellNeighs', **X})


def multitumor():
    S, P, U, X = from_quant('multitumor')
    write('multitumor', S, P, U, {'panel': f'all {len(P)} protein channels', 'source': 'TCIA CRC_FFPE-CODEX_CellNeighs', **X})


def bc_crc():
    S, P, U, X = from_quant('bc_crc')
    write('bc_crc', S, P, U, {'panel': f'{len(P)} protein channels included in both panels', 'source': 'Matusiak et al. 2024',
                              'tissue': {s: s.split('_')[0] for s in S}, **X})


def hnscc_ln():
    clin = pd.read_csv(f'{raw_dir("hnscc_ln")}/metadata/HNSCCdiscovery_clinmetadata.csv')
    pat = {r.unique_region: f'C{int(r.CaseID)}' for r in clin.itertuples()}
    S, P, U, X = from_quant('hnscc_ln', unit=pat.get)
    write('hnscc_ln', S, P, U, {'panel': f'all {len(P)} protein channels', 'source': 'S-BIAD1638 HNSCC/CODEX', **X})


def glioma():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import glioma as GL
    M = GL.sample_table().drop_duplicates('SampleID')
    pat = dict(zip(M.SampleID, M.PatientID))
    S, P, U, X = from_quant('glioma', qc_key='dapi_nuc_gt_ring', unit=pat.get)
    write('glioma', S, P, U, {'panel': f'{len(P)} markers shared by all sections (DAPI, IDHm excluded)',
                              'source': 'S-BIAD2840', 'registration': 'cellplex/preprocess/register.py', **X})


def bone_marrow():
    d = cohort_dir('bone_marrow')
    proteins = list(pd.read_csv(f'{PANELS}/bone_marrow.csv').marker)        # 52: both panels, no nuclear stain
    samples, qc, dropped = {}, {}, {}
    for f in sorted(os.listdir(f'{d}/quant')):
        Z = np.load(f'{d}/quant/{f}', allow_pickle=True)
        q = float(Z['dapi_nuc_gt_ring'])
        if q < MIN_QC:
            dropped[f[:-4]] = round(q, 3)
            continue
        names = [str(c) for c in Z['channels']]
        idx = [names.index(p) for p in proteins]
        samples[f[:-4]] = (Z['cx'].astype(np.float64), Z['cy'].astype(np.float64), np.log1p(Z['mean'][:, idx].astype(np.float64)),
                           np.arange(len(Z['cx'])))
        qc[f[:-4]] = round(q, 3)
    write('bone_marrow', samples, proteins, {s: s for s in samples},
          {'panel': '52 proteins shared by panels A (NPM1c) and B (CD19), DAPI excluded', 'source': 'Bandyopadhyay et al. 2024',
           'qc_nucleus_gt_ring': qc, 'excluded': dropped, 'exclusion': EXCLUSION})


if __name__ == '__main__':
    F = {'orion_crc': orion_crc, 'crc_tma': crc_tma, 'multitumor': multitumor, 'bc_crc': bc_crc, 'hnscc_ln': hnscc_ln,
         'bone_marrow': bone_marrow, 'glioma': glioma}
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True, choices=list(F))
    F[ap.parse_args().cohort]()
