"""Orion-CRC (Lin et al. 2023, Nat Cancer): H&E and 19-channel Orion immunofluorescence of the same section, H&E
registered to the Orion image by the authors (both 0.325 um/px), 41 slides / 40 patients -> CellViT++ nuclei and
per-cell protein truth.

Raw layout ($CELLPLEX_RAW/orion_crc/, as in s3://lin-2023-orion-crc/data/; data/urls/orion_crc.txt):
  {CRC01 .. CRC40, CRC33_01, CRC33_02}/*-registered.ome.tif   H&E, RGB uint8
                                       *-zlib.ome.tiff        Orion, (19, H, W) uint16
                                       markers.csv            channel names (Argo550 is a blank channel, dropped)
Slides are named CRC01_Scan1 .. CRC40_Scan1; CRC33_01 / CRC33_02 are CRC33_Scan1 / CRC33_Scan2 (one patient).

Steps (outputs under $CELLPLEX_DATA/orion_crc/):
  grid    the 512 px tiles of the slide that hold tissue (H&E saturation > 20 in >= 0.5% of the tile, measured on the
          4x-downsampled level)                                            -> grid/{slide}.npy (ny, nx) bool, .json
  wsi     H&E as a pyramidal JPEG TIFF on that grid, non-tissue tiles white (model input, 0.325 um/px)
                                                                           -> wsi/{slide}.tif, .json
  cvin    the same H&E resampled to 0.25 um/px for CellViT++ (bilinear, x_in = (x_out + 0.5) / sx - 0.5); with
          --pieces, slides larger than --budget tissue tiles are cut into horizontal strips that overlap by one
          512 px row (CellViT++ keeps every cell in memory)               -> wsi_cv/ or wsi_cv_pieces/
  (CellViT++: scripts/preprocess/run_cellvit.py process_wsi on each wsi_cv image, outputs seg_raw/{name}/)
  seg     CellViT++ cells.json -> nuclei in 0.325 um/px px (strips merged: each nucleus from the strip whose core
          rows hold its centroid)                -> seg/{slide}.parquet and seg/{slide}_cells.json (contour, centroid
          in level-0 px, parquet order; read by the specialist fine-tuning)
  quant   cell region = nucleus expanded by E um (0, 2, 4) without entering neighbours; value = mean Orion intensity
          in the region for all 18 channels (Hoechst and AF1 for QC)       -> quant/{slide}.npz

  python scripts/preprocess/orion_crc.py grid --slide CRC03_Scan1
  python scripts/preprocess/orion_crc.py wsi --slide CRC03_Scan1 --workers 16
  python scripts/preprocess/orion_crc.py cvin --slide CRC03_Scan1 --workers 16 [--pieces --budget 3000]
  python scripts/preprocess/run_cellvit.py --model $CELLVIT_DIR/checkpoints/CellViT-SAM-H-x40-AMP.pth \
      --outdir $CELLPLEX_DATA/orion_crc/seg_raw/CRC03_Scan1 process_wsi \
      --wsi_path $CELLPLEX_DATA/orion_crc/wsi_cv/CRC03_Scan1.tif --wsi_properties '{"slide_mpp": 0.25, "magnification": 40}'
  python scripts/preprocess/orion_crc.py seg --slide CRC03_Scan1
  python scripts/preprocess/orion_crc.py quant --slide CRC03_Scan1 --workers 16
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import tifffile
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cellplex.paths import cohort_dir, raw_dir  # noqa: E402
from cellplex.preprocess.cells import Cells, tile_index  # noqa: E402

RAW, D = raw_dir('orion_crc'), cohort_dir('orion_crc')
MPP, CV_MPP, TILE = 0.325, 0.25, 512
MARKERS = ['Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3', 'CD8', 'CD45RO', 'CD20',
           'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA']    # Orion channels without Argo550
PROTEINS = [m for m in MARKERS if m not in ('Hoechst', 'AF1')]                     # the 16 prediction targets
MIN_TISSUE = 0.005


def slides():
    """{slide: raw folder}"""
    out = {}
    for d in sorted(os.listdir(RAW)):
        if os.path.isdir(f'{RAW}/{d}'):
            out[f'CRC33_Scan{int(d[-1])}' if d.startswith('CRC33_') else f'{d}_Scan1'] = f'{RAW}/{d}'
    return out


def raw_files(slide):
    d = slides()[slide]
    he = glob.glob(f'{d}/*-registered.ome.tif')
    orion = glob.glob(f'{d}/*-zlib.ome.tiff')
    assert len(he) == 1 and len(orion) == 1, d
    names = [l.strip() for l in open(f'{d}/markers.csv') if l.strip() and l.strip() != 'marker_name']
    keep = [i for i, n in enumerate(names) if n != 'Argo550']
    assert len(keep) == len(MARKERS), names
    return he[0], orion[0], keep


G = {}                               # per-process image handles; set before pools fork


def _z(path, kind):
    if (path, kind) not in G:
        G[(path, kind)] = zarr.open(tifffile.imread(path, aszarr=True, level=0), mode='r')
    return G[(path, kind)]


def read_he(slide, iy, ix):
    """(512, 512, 3) uint8 H&E of grid tile (iy, ix), white outside the image."""
    he, _, _ = raw_files(slide)
    z = _z(he, 'he')
    H, W = z.shape[:2]
    y0, x0 = iy * TILE, ix * TILE
    out = np.full((TILE, TILE, 3), 255, np.uint8)
    a = np.asarray(z[y0:min(y0 + TILE, H), x0:min(x0 + TILE, W)])[..., :3]
    out[:a.shape[0], :a.shape[1]] = a
    return out


def read_orion(slide, iy, ix):
    """(18, 512, 512) uint16 Orion channels (MARKERS order) of grid tile (iy, ix), 0 outside the image."""
    _, orion, keep = raw_files(slide)
    z = _z(orion, 'orion')
    _, H, W = z.shape
    y0, x0 = iy * TILE, ix * TILE
    out = np.zeros((len(keep), TILE, TILE), np.uint16)
    a = np.asarray(z[:, y0:min(y0 + TILE, H), x0:min(x0 + TILE, W)])[keep]
    out[:, :a.shape[1], :a.shape[2]] = a
    return out


def grid(slide):
    return np.load(f'{D}/grid/{slide}.npy')


def tiles(slide):
    """{(iy, ix)} tissue tiles"""
    return {tuple(int(v) for v in t) for t in np.argwhere(grid(slide))}


# ----------------------------------------------------------------------------------------------------------- grid
def make_grid(slide):
    he, _, _ = raw_files(slide)
    with tifffile.TiffFile(he) as t:
        lv = t.series[0].levels
        H, W = lv[0].shape[:2]
        k = min(2, len(lv) - 1)
        a = lv[k].asarray()[..., :3]
    f = H / a.shape[0]
    sat = cv2.blur(cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1], (3, 3)) > 20
    ny, nx = int(np.ceil(H / TILE)), int(np.ceil(W / TILE))
    occ = np.zeros((ny, nx), bool)
    T = TILE / f
    for iy in range(ny):
        for ix in range(nx):
            s = sat[int(iy * T):int((iy + 1) * T), int(ix * T):int((ix + 1) * T)]
            occ[iy, ix] = s.size > 0 and s.mean() >= MIN_TISSUE
    iy, ix = np.nonzero(occ)
    occ = occ[:iy.max() + 1, :ix.max() + 1]                                # grid extent = last tissue tile
    os.makedirs(f'{D}/grid', exist_ok=True)
    np.save(f'{D}/grid/{slide}.npy', occ)
    json.dump({'slide': slide, 'image_W': W, 'image_H': H, 'ny': occ.shape[0], 'nx': occ.shape[1],
               'n_tiles': int(occ.sum()), 'min_tissue': MIN_TISSUE}, open(f'{D}/grid/{slide}.json', 'w'), indent=1)
    print(f'{slide}: {occ.sum()} tissue tiles of {occ.size}', flush=True)


# ------------------------------------------------------------------------------------------------------------ wsi
def _he_task(t):
    slide, iy, ix = t
    return read_he(slide, iy, ix)


def build_wsi(slide, workers, quality=90, min_px=1024):
    """Pyramid levels are separate pages of the main IFD chain (OpenSlide's generic TIFF reader), each 2x INTER_AREA."""
    t0 = time.time()
    occ = grid(slide)
    ny, nx = occ.shape
    H, W = ny * TILE, nx * TILE
    h2 = TILE // 2
    L1 = np.full((H // 2, W // 2, 3), 255, np.uint8)
    white = np.full((TILE, TILE, 3), 255, np.uint8)

    def level0(pool):
        for iy in range(ny):
            todo = [(slide, iy, ix) for ix in range(nx) if occ[iy, ix]]
            got = dict(zip([t[2] for t in todo], pool.map(_he_task, todo)))
            for ix in range(nx):
                im = got.get(ix)
                if im is None:
                    yield white
                    continue
                L1[iy * h2:(iy + 1) * h2, ix * h2:(ix + 1) * h2] = cv2.resize(im, (h2, h2), interpolation=cv2.INTER_AREA)
                yield im
            if iy % 20 == 0:
                print(f'  row {iy}/{ny} ({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(f'{D}/wsi', exist_ok=True)
    out, tmp = f'{D}/wsi/{slide}.tif', f'{D}/wsi/{slide}.tmp.tif'
    opts = dict(tile=(TILE, TILE), photometric='rgb', compression='jpeg', compressionargs={'level': quality},
                resolutionunit='CENTIMETER', maxworkers=workers)
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw, Pool(workers) as pool:
        tw.write(level0(pool), shape=(H, W, 3), dtype=np.uint8, resolution=(1e4 / MPP, 1e4 / MPP),
                 description=f'orion_crc {slide} mpp={MPP}', **opts)
        lv, k = L1, 1
        while True:
            tw.write(lv, subfiletype=1, resolution=(1e4 / (MPP * 2 ** k),) * 2, **opts)
            if min(lv.shape[:2]) < min_px:
                break
            lv = cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA)
            k += 1
    os.replace(tmp, out)
    json.dump({'slide': slide, 'W': W, 'H': H, 'mpp': MPP, 'tile': TILE, 'n_tiles': int(occ.sum()), 'levels': k + 1,
               'jpeg_quality': quality}, open(f'{D}/wsi/{slide}.json', 'w'), indent=1)
    print(f'{slide} done: {k + 1} levels, {os.path.getsize(out) / 1e9:.2f} GB ({time.time() - t0:.0f}s)', flush=True)


# ----------------------------------------------------------------------------------------------- CellViT++ input
def lin_idx(n_out, s, n_in):
    """For output pixels 0..n_out-1: the two source pixels and the source coordinate."""
    x = np.clip((np.arange(n_out) + 0.5) / s - 0.5, 0, n_in - 1)
    i0 = np.floor(x).astype(np.int64)
    return i0, np.minimum(i0 + 1, n_in - 1), x


def geometry(slide):
    occ = grid(slide)
    ny, nx = occ.shape
    H, W = ny * TILE, nx * TILE
    Wo, Ho = int(round(W * MPP / CV_MPP)), int(round(H * MPP / CV_MPP))
    Wo, Ho = Wo - Wo % TILE, Ho - Ho % TILE                              # whole output tiles
    return occ, W, H, Wo, Ho


def plan_pieces(slide, budget, ov_rows=1):
    """Horizontal strips of at most ~budget tissue tiles: [(rows0, rows1, core0, core1)] in output tile rows."""
    occ, W, H, Wo, Ho = geometry(slide)
    sy, nyo, ny = Ho / H, Ho // TILE, H // TILE
    cum = np.r_[0, np.cumsum(occ.sum(1))]
    y_in = np.arange(nyo + 1) * TILE / sy / TILE
    per_out = np.diff(np.interp(y_in, np.arange(ny + 1), cum))
    if per_out.sum() <= budget:
        return [(0, nyo, 0, nyo)]
    cuts, acc = [0], 0.0
    for ro, v in enumerate(per_out):
        if acc + v > budget and ro > cuts[-1]:
            cuts.append(ro); acc = 0.0
        acc += v
    cuts.append(nyo)
    return [(max(a - ov_rows, 0), min(b + ov_rows, nyo), a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def build_cvin(slide, workers, quality=90, out_dir=None, rows=None, name=None, core=None, min_px=1024):
    t0 = time.time()
    out_dir = out_dir or f'{D}/wsi_cv'
    occ, W, H, Wo, Ho = geometry(slide)
    nx = W // TILE
    sx, sy = Wo / W, Ho / H
    nyo, nxo = Ho // TILE, Wo // TILE
    ro0, ro1 = rows or (0, nyo)
    name = name or slide
    Hp = (ro1 - ro0) * TILE
    print(f'{name}: {W}x{H} @ {MPP} -> {Wo}x{Ho} @ ~{CV_MPP} (sx {sx:.6f}, sy {sy:.6f}), rows {ro0}..{ro1}', flush=True)
    xi0, xi1, xmap = lin_idx(Wo, sx, W)
    yi0, yi1, ymap = lin_idx(Ho, sy, H)
    h2 = TILE // 2
    L1 = np.full((Hp // 2, Wo // 2, 3), 255, np.uint8)
    white = np.full((TILE, TILE, 3), 255, np.uint8)
    cache = {}

    def in_row(pool, r):
        if r not in cache:
            todo = [(slide, r, c) for c in range(nx) if r < occ.shape[0] and occ[r, c]]
            if not todo:
                cache[r] = None
            else:
                got = dict(zip([t[2] for t in todo], pool.map(_he_task, todo)))
                cache[r] = np.concatenate([got.get(c, white) for c in range(nx)], 1)
        return cache[r]

    def level0(pool):
        for ro in range(ro0, ro1):
            ys = slice(ro * TILE, (ro + 1) * TILE)
            r0, r1 = yi0[ys].min() // TILE, yi1[ys].max() // TILE
            for k in list(cache):
                if k < r0:
                    del cache[k]
            rws = [in_row(pool, r) for r in range(r0, r1 + 1)]
            if all(r is None for r in rws):
                for _ in range(nxo):
                    yield white
                continue
            strip = np.concatenate([r if r is not None else np.full((TILE, W, 3), 255, np.uint8) for r in rws], 0)
            my = np.broadcast_to((ymap[ys] - r0 * TILE)[:, None], (TILE, TILE)).astype(np.float32)
            for co in range(nxo):
                xs = slice(co * TILE, (co + 1) * TILE)
                c0, c1 = xi0[xs].min(), xi1[xs].max() + 1
                src = strip[:, c0:c1]
                if src.min() >= 250:
                    yield white
                    continue
                mx = np.broadcast_to((xmap[xs] - c0)[None, :], (TILE, TILE)).astype(np.float32)
                t = cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if t.min() < 250:
                    L1[(ro - ro0) * h2:(ro - ro0 + 1) * h2, co * h2:(co + 1) * h2] = cv2.resize(
                        t, (h2, h2), interpolation=cv2.INTER_AREA)
                yield t
            if ro % 20 == 0:
                print(f'  row {ro}/{ro1} ({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out, tmp = f'{out_dir}/{name}.tif', f'{out_dir}/{name}.tmp.tif'
    mpp_x, mpp_y = MPP / sx, MPP / sy
    opts = dict(tile=(TILE, TILE), photometric='rgb', compression='jpeg', compressionargs={'level': quality},
                resolutionunit='CENTIMETER', maxworkers=workers)
    with tifffile.TiffWriter(tmp, bigtiff=True) as tw, Pool(workers) as pool:
        tw.write(level0(pool), shape=(Hp, Wo, 3), dtype=np.uint8, resolution=(1e4 / mpp_x, 1e4 / mpp_y),
                 description=f'orion_crc cellvit input {name}', **opts)
        lv, k = L1, 1
        while True:
            tw.write(lv, subfiletype=1, resolution=(1e4 / (mpp_x * 2 ** k), 1e4 / (mpp_y * 2 ** k)), **opts)
            if min(lv.shape[:2]) < min_px:
                break
            lv = cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA)
            k += 1
    os.replace(tmp, out)
    core = core or (ro0, ro1)
    json.dump({'slide': slide, 'name': name, 'W_in': W, 'H_in': H, 'W': Wo, 'H': Ho, 'sx': sx, 'sy': sy,
               'oy': ro0 * TILE, 'core_y': [core[0] * TILE, core[1] * TILE], 'mpp_in': MPP, 'mpp_x': mpp_x,
               'mpp_y': mpp_y, 'levels': k + 1, 'jpeg_quality': quality,
               'map_back': 'x_in = (x + 0.5) / sx - 0.5, y_in = (y + oy + 0.5) / sy - 0.5'},
              open(f'{out_dir}/{name}.json', 'w'), indent=1)
    print(f'{name} done: {os.path.getsize(out) / 1e9:.2f} GB ({time.time() - t0:.0f}s)', flush=True)


def build_pieces(slide, workers, budget):
    plan = plan_pieces(slide, budget)
    pd_ = f'{D}/wsi_cv_pieces'
    os.makedirs(pd_, exist_ok=True)
    names = [slide] if len(plan) == 1 else [f'{slide}__p{k}' for k in range(len(plan))]
    if len(plan) == 1:
        build_cvin(slide, workers)
    for nm, (a, b, c0, c1) in zip(names, plan):
        if len(plan) > 1 and not os.path.exists(f'{pd_}/{nm}.json'):
            build_cvin(slide, workers, out_dir=pd_, rows=(a, b), name=nm, core=(c0, c1))
    json.dump({'slide': slide, 'budget_tiles': budget,
               'pieces': [{'name': nm, 'rows': [a, b], 'core_rows': [c0, c1]} for nm, (a, b, c0, c1) in zip(names, plan)]},
              open(f'{pd_}/{slide}.plan.json', 'w'), indent=1)
    print(f'{slide}: {len(plan)} piece(s) {[(a, b) for a, b, _, _ in plan]}', flush=True)


# ------------------------------------------------------------------------------------------------------------ seg
def pieces(slide):
    """[(name, image json)] of the CellViT++ runs that make up one slide."""
    plan = f'{D}/wsi_cv_pieces/{slide}.plan.json'
    if os.path.exists(plan):
        P = json.load(open(plan))['pieces']
        if len(P) > 1:
            return [(p['name'], json.load(open(f'{D}/wsi_cv_pieces/{p["name"]}.json'))) for p in P]
    return [(slide, json.load(open(f'{D}/wsi_cv/{slide}.json')))]


def convert(slide):
    """CellViT JSON conventions: centroid [x, y], contour [[x, y], ...], bbox [[y0, x0], [y1, x1]]."""
    import shapely
    t0 = time.time()
    rings, cen, cls, n_bad, md, tm, per = [], [], [], 0, None, None, []
    for name, M in pieces(slide):
        sx, sy = M['sx'], M['sy']
        oy = M.get('oy', 0)
        lo, hi = M.get('core_y', [0, M['H']])
        with open(f'{D}/seg_raw/{name}/{name}_cells.json') as f:
            d = json.load(f)
        md = d['wsi_metadata']
        assert abs(md['base_mpp'] - md['target_patch_mpp']) < 0.035, f'CellViT rescaled internally: {md}'
        tmap = {int(k): v for k, v in d['type_map'].items()}
        tm = d['type_map']
        sc = np.array([sx, sy])
        kept = 0
        for c in d['cells']:
            if not lo <= c['centroid'][1] + oy < hi:                         # owned by the neighbouring strip
                continue
            r = np.asarray(c['contour'], np.float64)
            if len(r) < 3:
                (y0, x0), (y1, x1) = c['bbox']
                r = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
                n_bad += 1
            r = (r + (0.5, oy + 0.5)) / sc - 0.5
            if not np.array_equal(r[0], r[-1]):
                r = np.vstack([r, r[:1]])
            rings.append(r)
            cen.append((np.asarray(c['centroid'], np.float64) + (0.5, oy + 0.5)) / sc - 0.5)
            cls.append(tmap[c['type']])
            kept += 1
        per.append({'name': name, 'detected': len(d['cells']), 'kept': kept})
        del d
    geom = shapely.polygons(shapely.linearrings(
        np.concatenate(rings), indices=np.repeat(np.arange(len(rings)), [len(r) for r in rings])))
    cen = np.asarray(cen)
    df = pd.DataFrame({'geometry': shapely.to_wkb(geom), 'class': pd.Categorical(cls),
                       'cell_id': np.arange(len(rings), dtype=np.int64),
                       'cx': cen[:, 0].astype(np.float32), 'cy': cen[:, 1].astype(np.float32)})
    os.makedirs(f'{D}/seg', exist_ok=True)
    tmp = f'{D}/seg/{slide}.tmp.parquet'
    df.to_parquet(tmp, compression='zstd', index=False)
    os.replace(tmp, f'{D}/seg/{slide}.parquet')
    json.dump({'wsi_metadata': md, 'type_map': tm, 'n_cells': len(rings), 'n_degenerate': n_bad, 'pieces': per},
              open(f'{D}/seg/{slide}.meta.json', 'w'), indent=1)
    print(f'{slide}: {len(rings):,} nuclei from {len(per)} CellViT run(s) -> parquet, degenerate {n_bad} '
          f'({time.time() - t0:.0f}s)', flush=True)
    cells_json(slide)


def cells_json(slide):
    """seg/{slide}.parquet -> seg/{slide}_cells.json: {'cells': [{'contour': [[x, y], ...], 'centroid': [x, y]}]} in
    level-0 px and parquet order (the layout of the CellViT++ output of the other whole-slide cohorts)."""
    import pyarrow.parquet as pq
    import shapely
    t = pq.read_table(f'{D}/seg/{slide}.parquet', columns=['geometry', 'cx', 'cy'])
    geom = shapely.from_wkb(t.column('geometry').to_numpy(zero_copy_only=False))
    xy, gid = shapely.get_coordinates(geom, return_index=True)
    ptr = np.r_[0, np.cumsum(np.bincount(gid, minlength=len(geom)))]
    cx, cy = t.column('cx').to_numpy().tolist(), t.column('cy').to_numpy().tolist()
    tmp = f'{D}/seg/{slide}_cells.tmp.json'
    with open(tmp, 'w') as fh:
        json.dump({'cells': [{'contour': xy[ptr[i]:ptr[i + 1]].tolist(), 'centroid': [cx[i], cy[i]]}
                             for i in range(len(geom))], 'coordinates': 'level-0 px, 0.325 um/px'}, fh)
    os.replace(tmp, f'{D}/seg/{slide}_cells.json')
    print(f'{slide}: {len(geom):,} nuclei -> seg/{slide}_cells.json', flush=True)


# ---------------------------------------------------------------------------------------------------------- quant
def _quant_task(task):
    from skimage.segmentation import expand_labels
    slide, iy, ix = task
    C, M, E = G['cells'], G['margin'], G['exp_px']
    idx = G['tidx'].get((iy, ix))
    if idx is None:
        return None
    n = len(idx)
    S = TILE + 2 * M
    lab0 = C.draw(idx, ix * TILE - M, iy * TILE - M, (S, S))
    cod = read_orion(slide, iy, ix).reshape(len(MARKERS), -1).astype(np.float64)
    dark = 255.0 - read_he(slide, iy, ix).transpose(2, 0, 1).reshape(3, -1).mean(0)   # H&E darkness, segmentation QC
    cnt = np.zeros((n, len(E)), np.int64)
    sums = np.zeros((n, len(E), len(MARKERS) + 1), np.float64)
    for k, e in enumerate(E):
        lab = lab0 if e == 0 else expand_labels(lab0, distance=e)
        l = lab[M:M + TILE, M:M + TILE].ravel()
        cnt[:, k] = np.bincount(l, minlength=n + 1)[1:]
        for c in range(len(MARKERS)):
            sums[:, k, c] = np.bincount(l, weights=cod[c], minlength=n + 1)[1:]
        sums[:, k, -1] = np.bincount(l, weights=dark, minlength=n + 1)[1:]
    keep = cnt.any(1)
    return idx[keep], cnt[keep], sums[keep]


def quant(slide, expand_um, workers):
    """Regions crossing tile borders are summed over every tile they touch; nuclei within a margin are drawn on a
    (512 + 2m) canvas before expansion, so expansion near a tile edge sees the same neighbours as on the slide."""
    t0 = time.time()
    C = Cells(f'{D}/seg/{slide}.parquet')
    exp_px = [e / MPP for e in expand_um]
    M = int(np.ceil(max(exp_px))) + 2
    tasks = [(slide, iy, ix) for iy, ix in sorted(tiles(slide))]
    ny, nx = max(t[1] for t in tasks) + 2, max(t[2] for t in tasks) + 2
    G.update(cells=C, margin=M, exp_px=exp_px, tidx=tile_index(C.bb, TILE, M, ny, nx))
    print(f'{slide}: {C.n:,} nuclei, {len(tasks)} tiles, expand {expand_um} um, margin {M} px', flush=True)
    K = len(exp_px)
    AREA = np.zeros((C.n, K), np.int64)
    SUM = np.zeros((C.n, K, len(MARKERS) + 1), np.float64)
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_quant_task, tasks, chunksize=8)):
            if r is not None:
                idx, cnt, sums = r
                AREA[idx] += cnt
                SUM[idx] += sums
            if i % 2000 == 0:
                print(f'  {i}/{len(tasks)} tiles ({time.time() - t0:.0f}s)', flush=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = (SUM / AREA[..., None]).astype(np.float32)
    mean[AREA == 0] = np.nan
    mean, he_dark = mean[..., :-1], mean[..., -1]
    if 0 in expand_um and K > 1:                       # nucleus darker in H&E / brighter in Hoechst than its ring
        k0, k1 = expand_um.index(0), 1 if expand_um.index(0) == 0 else 0
        ring_n = AREA[:, k1] - AREA[:, k0]
        ok = (AREA[:, k0] > 0) & (ring_n > 0)
        for name, c in (('H&E darkness', -1), ('Hoechst', MARKERS.index('Hoechst'))):
            ring = (SUM[:, k1, c] - SUM[:, k0, c]) / np.maximum(ring_n, 1)
            nuc = SUM[:, k0, c] / np.maximum(AREA[:, k0], 1)
            print(f'  QC {name}: nucleus > ring in {np.mean(nuc[ok] > ring[ok]):.1%} of cells', flush=True)
    os.makedirs(f'{D}/quant', exist_ok=True)
    tmp = f'{D}/quant/{slide}.tmp.npz'
    np.savez(tmp, cell_id=C.cid, cx=C.cx, cy=C.cy, cls=C.cls, markers=np.array(MARKERS),
             expand_um=np.array(expand_um, np.float32), area=AREA.astype(np.int32), mean=mean,
             he_dark=he_dark.astype(np.float32))
    os.replace(tmp, f'{D}/quant/{slide}.npz')
    print(f'{slide} done ({time.time() - t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['grid', 'wsi', 'cvin', 'seg', 'json', 'quant'])
    ap.add_argument('--slide', default='', help='comma-separated; default: every slide of the raw folder')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pieces', action='store_true', help='cvin: strips of <= --budget tissue tiles for large slides')
    ap.add_argument('--budget', type=int, default=3000)
    ap.add_argument('--expand_um', type=float, nargs='+', default=[0.0, 2.0, 4.0])
    a = ap.parse_args()
    for s in (a.slide.split(',') if a.slide else list(slides())):
        if a.step == 'grid':
            make_grid(s)
        elif a.step == 'wsi':
            build_wsi(s, a.workers)
        elif a.step == 'cvin':
            build_pieces(s, a.workers, a.budget) if a.pieces else build_cvin(s, a.workers)
        elif a.step == 'seg':
            convert(s)
        elif a.step == 'json':
            cells_json(s)
        else:
            quant(s, a.expand_um, a.workers)


if __name__ == '__main__':
    main()
