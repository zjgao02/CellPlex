"""Pixel-level targets for fine-tuning the H&E -> multiplex models (Eva, GigaTIME) on a processed cohort.

Every panel protein (meta.json 'proteins') from the multiplex image, moved into the H&E frame with the H&E ->
multiplex shift used for the truth (multiplex = H&E + (dy, dx)), clipped to the sample's p0.5..p99.5 over tissue
pixels and scaled to [0, 1] (stored as uint8) at the H&E's native resolution; tissue = H&E pixel with saturation > 20
after a 5 px blur and inside the shifted multiplex field.

  crc_tma, multitumor, bc_crc   per core / region -> pixel/{sample}.npz (t (P, H, W), mask, lo, hi)
  hnscc_ln                      memory-mapped: pixel/{sample}_t.npy (H, W, P), pixel/{sample}_mask.npy, _meta.npz
  orion_crc, bone_marrow,       whole slides: only the per-sample normalisation, from 256 random 256 px native tissue
  glioma                        crops, and a 1/16 tissue mask of window centres -> pixel/{sample}_norm.npz (the
                                fine-tuning reads H&E and multiplex crops on the fly; glioma: through the registration;
                                orion_crc: window centres kept >= EDGE px inside the Orion image, which the H&E grid
                                of wsi/ can overhang by part of a 512 px tile)

  python scripts/preprocess/pixel_targets.py --cohort multitumor
"""
import argparse, json, os, sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from cellplex.paths import cohort_dir  # noqa: E402
from cellplex.preprocess.common import tissue_of  # noqa: E402

EDGE = 1024                     # orion_crc: window centres at least this far (native px) from the Orion image's far edges


def shift_img(a, dy, dx):
    """Multiplex frame -> H&E frame: out(y, x) = a(y + dy, x + dx); outside -> 0 and invalid."""
    H, W = a.shape[-2:]
    out = np.zeros_like(a)
    valid = np.zeros((H, W), bool)
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[..., yd, xd] = a[..., ys, xs]
    valid[yd, xd] = True
    return out, valid


def scale(x, tissue):
    """(P, H, W) -> uint8 [0, 255] by per-channel p0.5..p99.5 over tissue; also lo, hi."""
    P = x.shape[0]
    t = np.zeros(x.shape, np.uint8)
    lo, hi = np.zeros(P, np.float32), np.zeros(P, np.float32)
    for j in range(P):
        v = x[j][tissue].astype(np.float32) if tissue.any() else x[j].ravel().astype(np.float32)
        lo[j], hi[j] = np.percentile(v, 0.5), np.percentile(v, 99.5)
        t[j] = np.round(np.clip((x[j].astype(np.float32) - lo[j]) / max(hi[j] - lo[j], 1e-6), 0, 1) * 255).astype(np.uint8)
    return t, lo, hi


def sample_ids(cohort):
    return sorted(f[:-4] for f in os.listdir(f'{cohort_dir(cohort)}/cells'))


def arrays(cohort, prots, out):
    """Cores / regions: the multiplex image shifted by the sample's shift (quant/{sample}.npz)."""
    d = cohort_dir(cohort)
    if cohort == 'bc_crc':
        import bc_crc as M
    elif cohort == 'hnscc_ln':
        import hnscc_ln as M
    else:
        import codex_tma as M
        S = M.samples(cohort)
    for s in sample_ids(cohort):
        dy, dx = (int(v) for v in np.load(f'{d}/quant/{s}.npz', allow_pickle=True)['shift'])
        if cohort == 'bc_crc':
            pidx, _ = M.panel(s)
            cod, ch = tifffile.imread(M.files(s)[1]), [pidx[p] for p in prots]
        elif cohort == 'hnscc_ln':
            _, pidx, _ = M.channels(s)
            cod, ch = M.read_codex(s), [pidx[p] for p in prots]
        else:
            tma, _, cp = S[s]
            cod = M.read_codex(cp)
            names = M.channel_names(cohort, tma)[:cod.shape[0]]
            ch = [names.index(p) for p in prots]
        x, valid = shift_img(cod[ch], dy, dx)
        tissue = tissue_of(np.load(f'{d}/he/{s}.npy')) & valid
        t, lo, hi = scale(x, tissue)
        if cohort == 'hnscc_ln':                                           # 388 x 57 x 1440 x 1920: memory-mapped
            np.save(f'{out}/{s}_t.npy', np.ascontiguousarray(t.transpose(1, 2, 0)))
            np.save(f'{out}/{s}_mask.npy', tissue)
            np.savez(f'{out}/{s}_meta.npz', lo=lo, hi=hi, proteins=np.array(prots), shift=np.array([dy, dx]))
        else:
            np.savez_compressed(f'{out}/{s}.npz', t=t, mask=tissue, lo=lo, hi=hi, proteins=np.array(prots),
                                shift=np.array([dy, dx]))
        print(f'{s}: {len(prots)} proteins, shift ({dy}, {dx}), tissue {tissue.mean():.1%}', flush=True)


def slides(cohort, prots, out, samples='', n_crop=256, L=256, seed=0):
    """Whole slides: per-sample p0.5..p99.5 from random native tissue crops; glioma samples the CODEX through the
    registration (reg/{sample}.npz), bone marrow reads the CODEX on the H&E grid."""
    import zarr
    from cellplex.preprocess.register import he_to_cx
    d = cohort_dir(cohort)
    meta = json.load(open(f'{d}/meta.json'))
    if cohort == 'glioma':
        import glioma as M
        F = M.files()
    elif cohort == 'orion_crc':
        import orion_crc as M
    else:
        import bone_marrow as M
    for s in (samples.split(',') if samples else sample_ids(cohort)):
        if cohort == 'glioma':
            cx_path = F[s][0]
            names = M.channel_names(cx_path)
            reg = dict(np.load(f'{d}/reg/{s}.npz'))
        elif cohort == 'orion_crc':
            _, cx_path, keep = M.raw_files(s)                              # 19-channel Orion image, Argo550 skipped
            names, reg = [None] * (max(keep) + 1), None
            for i, m in zip(keep, M.MARKERS):
                names[i] = m
        else:
            cx_path, names, reg = M.if_path(s, 'aligned'), M.channel_names(s), None
        ch = [names.index(p) for p in prots]
        with tifffile.TiffFile(meta['image'].format(sample=s)) as t:
            lv = [p for p in t.series[0].levels]
            k = next(i for i, l in enumerate(lv) if l.shape[0] <= lv[0].shape[0] // 16 + 1)
            f = lv[0].shape[0] / lv[k].shape[0]
            thumb = lv[k].asarray()[..., :3]
        z = zarr.open(tifffile.imread(cx_path, aszarr=True, level=0), mode='r')
        he = zarr.open(tifffile.imread(meta['image'].format(sample=s), aszarr=True, level=0), mode='r')
        H, W = he.shape[:2]
        low = tissue_of(thumb)
        if cohort == 'orion_crc':
            yy, xx = np.mgrid[0:thumb.shape[0], 0:thumb.shape[1]] * f
            low &= (xx <= z.shape[2] - EDGE) & (yy <= z.shape[1] - EDGE)
        if reg is not None:
            yy, xx = np.mgrid[0:thumb.shape[0], 0:thumb.shape[1]].astype(np.float32) * np.float32(f)
            mx, my = he_to_cx(reg, xx, yy)
            low &= (mx >= 0) & (my >= 0) & (mx <= z.shape[2] - 1) & (my <= z.shape[1] - 1)
        rng = np.random.default_rng(seed)
        pts = np.argwhere(low)
        vals = [[] for _ in ch]
        for y, x in pts[rng.choice(len(pts), min(n_crop, len(pts)), replace=False)]:
            y0 = int(np.clip(y * f - L // 2, 0, H - L)); x0 = int(np.clip(x * f - L // 2, 0, W - L))
            m = tissue_of(np.asarray(he[y0:y0 + L, x0:x0 + L])[..., :3])
            if reg is not None:
                a, inside = codex_on_he(z, reg, x0, y0, L, L, ch)
                m &= inside
                for j in range(len(ch)):
                    vals[j].append(a[j][m])
            else:
                a = np.asarray(z[:, y0:y0 + L, x0:x0 + L])
                for j, c in enumerate(ch):
                    vals[j].append(a[c][m])
        v = [np.concatenate(q).astype(np.float32) for q in vals]
        lo = np.array([np.percentile(q, 0.5) for q in v], np.float32); hi = np.array([np.percentile(q, 99.5) for q in v], np.float32)
        extra = {} if reg is None else dict(A=reg['A'], field=reg['field'], tile=reg['tile'])
        np.savez(f'{out}/{s}_norm.npz', lo=lo, hi=hi, channels=np.array(ch), proteins=np.array(prots), low=low, factor=f,
                 if_path=cx_path, n_tissue_px=len(v[0]), **extra)
        print(f'{s}: {len(prots)} proteins, multiplex {z.shape}, tissue (1/{f:.0f}) {low.mean():.1%}, '
              f'{len(v[0]):,} px for the percentiles', flush=True)


def codex_on_he(z, reg, x0, y0, w, h, chans):
    """Registered multiplex channels sampled bilinearly on the H&E pixel grid of a window -> (len(chans), h, w)
    float32 and the mask of pixels inside the multiplex image."""
    import cv2
    from cellplex.preprocess.register import he_to_cx
    C, HC, WC = z.shape
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    mx, my = he_to_cx(reg, xx, yy)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True,
                    choices=['orion_crc', 'crc_tma', 'multitumor', 'bc_crc', 'hnscc_ln', 'bone_marrow', 'glioma'])
    ap.add_argument('--samples', default='', help='whole-slide cohorts: only these samples (comma-separated)')
    a = ap.parse_args()
    prots = json.load(open(f'{cohort_dir(a.cohort)}/meta.json'))['proteins']
    out = f'{cohort_dir(a.cohort)}/pixel'
    os.makedirs(out, exist_ok=True)
    if a.cohort in ('orion_crc', 'bone_marrow', 'glioma'):
        slides(a.cohort, prots, out, a.samples)
    else:
        arrays(a.cohort, prots, out)


if __name__ == '__main__':
    main()
