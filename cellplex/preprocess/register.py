"""Registration of a whole-slide multiplex image (PhenoCycler Fusion, ~0.507 um/px) to the H&E of the same section
(stained after the multiplex imaging, ~0.2485 um/px).

Mapping, H&E level-0 px -> multiplex level-0 px:  c = A @ [x, y, 1] + d(A @ [x, y, 1])
  1. coarse (8 um/px): tissue masks, search over rotation (+-4 deg) and scale (+-3 %), translation by phase correlation
     (or, when the tissue fills the multiplex field, the same search on blurred hematoxylin vs log DAPI)
  2. affine (2 um/px): ECC (MOTION_AFFINE) of the H&E hematoxylin (rgb2hed) on log DAPI, unimaged areas masked
  3. local (multiplex native px): 512 px tiles, phase correlation of hematoxylin vs DAPI, outliers rejected,
     median-filtered, smoothed displacement field d on the tile grid (bilinear between tile centres)
QC per tile: Pearson r of hematoxylin vs DAPI (1 um blur) at the affine-only and the final mapping.
"""
import re

import cv2
import numpy as np
import tifffile
from skimage.color import rgb2hed

TILE = 512


def hema(he):
    h = rgb2hed(he)[..., 0].astype(np.float32)
    h[he.min(2) > 235] = h.min()                                       # white background
    return h


def norm(a, m=None):
    v = a[m] if m is not None and m.any() else a.ravel()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def at(img, mpp_in, mpp_out, interp=cv2.INTER_AREA):
    f = mpp_in / mpp_out
    return cv2.resize(img, (max(1, int(round(img.shape[1] * f))), max(1, int(round(img.shape[0] * f)))), interpolation=interp)


def _to_level0(M, mpp, mpp_he, mpp_cx):
    """Affine between `mpp` images -> H&E level-0 -> multiplex level-0."""
    S_in = np.diag([mpp / mpp_he, mpp / mpp_he, 1.0]); S_out = np.diag([mpp_cx / mpp, mpp_cx / mpp, 1.0])
    return np.linalg.inv(S_out) @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(S_in)


def coarse(he_lo, mpp_he_lo, dapi_lo, mpp_cx_lo, mpp_he, mpp_cx, mpp=8.0):
    """Affine H&E level-0 -> multiplex level-0 from tissue masks at `mpp`."""
    h = at(he_lo, mpp_he_lo, mpp); d = at(dapi_lo, mpp_cx_lo, mpp)
    mh = (cv2.blur(cv2.cvtColor(h, cv2.COLOR_RGB2HSV)[..., 1], (3, 3)) > 20).astype(np.float32)
    dl = norm(np.log1p(d.astype(np.float32)), d > 0)
    thr = cv2.threshold((dl * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255
    md = ((dl > thr) & (d > 0)).astype(np.float32)
    md = cv2.morphologyEx(md, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    H, W = max(mh.shape[0], md.shape[0]) + 64, max(mh.shape[1], md.shape[1]) + 64
    pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])))   # noqa: E731
    D = pad(md); win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    best = (-1, None)
    for ang in np.arange(-4, 4.01, 0.5):
        for sc in np.arange(0.97, 1.031, 0.01):
            M = cv2.getRotationMatrix2D((mh.shape[1] / 2, mh.shape[0] / 2), ang, sc)
            Hm = cv2.warpAffine(pad(mh), M, (W, H))
            (dx, dy), r = cv2.phaseCorrelate(Hm * win, D * win)
            M2 = M.copy(); M2[:, 2] += (dx, dy)
            ov = (cv2.warpAffine(pad(mh), M2, (W, H)) * D).sum() / max(np.sqrt(mh.sum() * md.sum()), 1)
            if ov > best[0]:
                best = (ov, M2, ang, sc)
    ov, M, ang, sc = best
    return _to_level0(M, mpp, mpp_he, mpp_cx), dict(overlap=float(ov), angle=float(ang), scale=float(sc))


def coarse_intensity(he_lo, mpp_he_lo, dapi_lo, mpp_cx_lo, mpp_he, mpp_cx, mpp=8.0):
    """When the tissue fills the multiplex field (mask edges = field edges): the same search on blurred hematoxylin
    vs log DAPI, zero-mean inside the imaged area."""
    h = at(he_lo, mpp_he_lo, mpp); d = at(dapi_lo, mpp_cx_lo, mpp)
    hm = cv2.GaussianBlur(norm(hema(h)), (0, 0), 2.0); valid = (d > 0).astype(np.float32)
    dm = cv2.GaussianBlur(norm(np.log1p(d.astype(np.float32)), d > 0), (0, 0), 2.0)
    dm = (dm - dm[valid > 0].mean()) * valid
    H, W = max(hm.shape[0], dm.shape[0]) + 64, max(hm.shape[1], dm.shape[1]) + 64
    pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])))   # noqa: E731
    D = pad(dm); best = (-np.inf, None)
    for ang in np.arange(-4, 4.01, 0.5):
        for sc in np.arange(0.97, 1.031, 0.01):
            M = cv2.getRotationMatrix2D((hm.shape[1] / 2, hm.shape[0] / 2), ang, sc)
            Hm = cv2.warpAffine(pad(hm), M, (W, H))
            Hc = (Hm - Hm[pad(valid) > 0].mean()) if (pad(valid) > 0).any() else Hm
            (dx, dy), r = cv2.phaseCorrelate(Hc, D)
            M2 = M.copy(); M2[:, 2] += (dx, dy)
            w = cv2.warpAffine(pad(hm), M2, (W, H)); v = pad(valid) > 0
            cc = np.corrcoef(w[v], D[v])[0, 1] if v.sum() > 100 else -1
            if cc > best[0]:
                best = (cc, M2, ang, sc)
    cc, M, ang, sc = best
    return _to_level0(M, mpp, mpp_he, mpp_cx), dict(overlap=float(cc), angle=float(ang), scale=float(sc))


def warp_he_to_cx(hmap, mpp_h, A, shape_cx, mpp_c, mpp_he, mpp_cx):
    """hmap (H&E frame, mpp_h) resampled onto the multiplex grid at mpp_c through A."""
    S_h = np.diag([mpp_he / mpp_h, mpp_he / mpp_h, 1.0]); S_c = np.diag([mpp_cx / mpp_c, mpp_cx / mpp_c, 1.0])
    B = S_c @ A @ np.linalg.inv(S_h)                                     # hmap px -> multiplex (mpp_c) px
    return cv2.warpAffine(hmap, B[:2], (shape_cx[1], shape_cx[0]), flags=cv2.INTER_LINEAR)


def ecc_affine(A, he_lv, mpp_h, dapi_lv, mpp_c, mpp_he, mpp_cx, mpp=2.0):
    h = norm(cv2.GaussianBlur(hema(at(he_lv, mpp_h, mpp)), (0, 0), 1.0))
    d0 = at(dapi_lv, mpp_c, mpp); valid = (d0 > 0).astype(np.uint8)
    d = norm(cv2.GaussianBlur(np.log1p(d0.astype(np.float32)), (0, 0), 1.0), valid > 0)
    hw = warp_he_to_cx(h, mpp, A, d.shape, mpp, mpp_he, mpp_cx)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(d, hw, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), valid, 5)
    except cv2.error as e:
        print('  ECC failed:', str(e)[:80]); return A
    # hw(x) = h(B x); ECC finds W with d(x) ~ hw(W x)  ->  A_new = inv(W') A
    Wm = np.vstack([warp, [0, 0, 1]])
    S_c = np.diag([mpp_cx / mpp, mpp_cx / mpp, 1.0])
    return np.linalg.inv(S_c) @ np.linalg.inv(Wm) @ S_c @ A


def local_field(A, he_path, dapi_full, valid_full, mpp_h1, mpp_he, mpp_cx):
    """Tile shifts (multiplex px) of hematoxylin vs DAPI at multiplex resolution, after A."""
    with tifffile.TiffFile(he_path) as t:
        he1 = t.series[0].levels[1].asarray()                            # ~0.497 um/px
    h = hema(he1)
    hw = warp_he_to_cx(h, mpp_h1, A, dapi_full.shape, mpp_cx, mpp_he, mpp_cx)
    d = np.log1p(dapi_full.astype(np.float32))
    ny, nx = int(np.ceil(dapi_full.shape[0] / TILE)), int(np.ceil(dapi_full.shape[1] / TILE))
    sh = np.full((ny, nx, 2), np.nan, np.float32); resp = np.zeros((ny, nx), np.float32)
    r0 = np.full((ny, nx), np.nan, np.float32)
    win = cv2.createHanningWindow((TILE, TILE), cv2.CV_32F)
    thr = np.percentile(hw[::8, ::8], 60)
    for i in range(ny):
        for j in range(nx):
            ys, xs = slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)
            a, b, v = hw[ys, xs], d[ys, xs], valid_full[ys, xs]
            if a.shape != (TILE, TILE) or v.mean() < 0.9 or (a > thr).mean() < 0.2:
                continue
            a = cv2.GaussianBlur(norm(a), (0, 0), 1.5); b = cv2.GaussianBlur(norm(b), (0, 0), 1.5)
            r0[i, j] = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            (dx, dy), r = cv2.phaseCorrelate(a * win, b * win)            # b(x) ~ a(x - shift)
            if r > 0.05 and abs(dx) < 40 and abs(dy) < 40:
                sh[i, j] = (dx, dy); resp[i, j] = r
    return sh, resp, r0, hw, d


def smooth_field(sh):
    f = sh.copy()
    ok = np.isfinite(f[..., 0])
    if not ok.any():
        return np.zeros_like(f)
    med = np.nanmedian(f[ok], 0)
    dev = np.linalg.norm(f - med, axis=2)
    f[~ok | (dev > 15)] = np.nan                                         # outliers -> filled from neighbours
    out = f.copy()
    for _ in range(50):                                                  # nearest-neighbour fill of the gaps
        miss = ~np.isfinite(out[..., 0])
        if not miss.any():
            break
        for c in range(2):
            ch = out[..., c]; fill = cv2.blur(np.nan_to_num(ch), (3, 3)); cnt = cv2.blur(np.isfinite(ch).astype(np.float32), (3, 3))
            ch[miss & (cnt > 0)] = (fill / np.maximum(cnt, 1e-6))[miss & (cnt > 0)]
    out = np.nan_to_num(out, nan=0.0)
    for c in range(2):
        out[..., c] = cv2.GaussianBlur(cv2.medianBlur(out[..., c].astype(np.float32), 3), (0, 0), 1.0)
    return out


def register(he_path, cx_path, dapi, mpp_he, mpp_cx, out_npz, qc_jpg=None, method='mask', log=print):
    """Full registration of one section -> out_npz (A, field, tile, QC); he_path: pyramidal H&E TIFF (level 1 ~0.5,
    level 3 ~2, level 5 ~8 um/px); cx_path: pyramidal multiplex TIFF, `dapi` its nuclear-stain channel."""
    import time
    t0 = time.time()
    with tifffile.TiffFile(he_path) as t:
        lv = t.series[0].levels; H0 = lv[0].shape
        he_lo = lv[5].asarray(); mpp_h5 = mpp_he * H0[0] / lv[5].shape[0]
        he_l2 = lv[3].asarray(); mpp_h3 = mpp_he * H0[0] / lv[3].shape[0]
        mpp_h1 = mpp_he * H0[0] / lv[1].shape[0]
    with tifffile.TiffFile(cx_path) as t:
        lv = t.series[0].levels; C0 = lv[0].shape[-2:]
        d_lo = lv[4].asarray()[dapi]; mpp_c4 = mpp_cx * C0[0] / d_lo.shape[0]
        d_l2 = lv[2].asarray()[dapi]; mpp_c2 = mpp_cx * C0[0] / d_l2.shape[0]
    dapi_full = tifffile.imread(cx_path, series=0, level=0, key=dapi)
    log(f'inputs read ({time.time() - t0:.0f}s)')
    A0, info = (coarse_intensity if method == 'intensity' else coarse)(he_lo, mpp_h5, d_lo, mpp_c4, mpp_he, mpp_cx)
    A1 = ecc_affine(A0, he_l2, mpp_h3, d_l2, mpp_c2, mpp_he, mpp_cx)
    log(f'affine done ({time.time() - t0:.0f}s)')
    sh, resp, r_aff, hw, d = local_field(A1, he_path, dapi_full, dapi_full > 0, mpp_h1, mpp_he, mpp_cx)
    field = smooth_field(sh)
    r_fin = np.full_like(r_aff, np.nan)                                  # re-measure tile r after the local shift
    for i in range(r_aff.shape[0]):
        for j in range(r_aff.shape[1]):
            if not np.isfinite(r_aff[i, j]):
                continue
            ys, xs = slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE)
            M = np.float32([[1, 0, field[i, j, 0]], [0, 1, field[i, j, 1]]])
            aa = cv2.warpAffine(hw[ys, xs], M, (TILE, TILE), borderMode=cv2.BORDER_REFLECT)
            aa = cv2.GaussianBlur(norm(aa), (0, 0), 1.5); bb = cv2.GaussianBlur(norm(d[ys, xs]), (0, 0), 1.5)
            r_fin[i, j] = np.corrcoef(aa.ravel(), bb.ravel())[0, 1]
    np.savez(out_npz, A=A1, A_coarse=A0, field=field, tile=TILE, shifts_raw=sh, resp=resp, r_affine=r_aff,
             r_final=r_fin, he_shape=np.array(H0[:2]), cx_shape=np.array(C0), mpp_he=mpp_he, mpp_cx=mpp_cx,
             dapi_channel=dapi, **{f'coarse_{kk}': v for kk, v in info.items()})
    q = np.isfinite(r_aff)
    mag = np.linalg.norm(field[q], axis=1) if q.any() else np.array([0])
    log(f'coarse overlap {info["overlap"]:.2f} (rot {info["angle"]:+.1f} deg, scale {info["scale"]:.2f}); {q.sum()} tiles; '
        f'tile r hema~DAPI affine {np.nanmedian(r_aff):.3f} -> local {np.nanmedian(r_fin):.3f}; local shift median '
        f'{np.median(mag):.1f} px, p90 {np.percentile(mag, 90):.1f} px ({time.time() - t0:.0f}s)')
    if qc_jpg and q.any():                    # 1 mm crop at the tissue centre: hematoxylin magenta, DAPI green
        yy, xx = np.argwhere(q).mean(0).astype(int) * TILE + TILE // 2
        L = 1000
        y0 = int(np.clip(yy - L, 0, d.shape[0] - 2 * L)); x0 = int(np.clip(xx - L, 0, d.shape[1] - 2 * L))

        def stretch(x):
            lo, hi = np.percentile(x, 50), np.percentile(x, 99.7)
            return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        a_ = stretch(cv2.GaussianBlur(hw[y0:y0 + 2 * L, x0:x0 + 2 * L], (0, 0), 1.0))
        b_ = stretch(cv2.GaussianBlur(d[y0:y0 + 2 * L, x0:x0 + 2 * L], (0, 0), 1.0))
        rgb = (np.stack([a_, b_, a_], -1) * 255).astype(np.uint8)
        cv2.imwrite(qc_jpg, cv2.cvtColor(cv2.resize(rgb, (1000, 1000)), cv2.COLOR_RGB2BGR))


def field_at(field, tile, cx, cy):
    """Local shift (multiplex px) at multiplex coordinates: bilinear between tile centres."""
    fx = (cx - tile / 2) / tile; fy = (cy - tile / 2) / tile
    out = []
    for c in range(2):
        out.append(cv2.remap(field[..., c].astype(np.float32), fx.astype(np.float32), fy.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def he_to_cx(reg, xx, yy):
    """Multiplex px (mx, my) of H&E px (xx, yy): affine, then the local field."""
    A = reg['A']
    ax = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]; ay = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    dx, dy = field_at(reg['field'], int(reg['tile']), ax, ay)
    return ax + dx, ay + dy


def sample_on_grid(reg_, z, cx_names, prots, polys, H, W, mpp_he, expand_um=2.0, block=4096, log=print):
    """Truth of every nucleus (H&E px polygons) from a registered multiplex image: per block, regions = nucleus + 2 um,
    each region pixel mapped through the registration and sampled bilinearly."""
    from skimage.segmentation import expand_labels
    idx = [cx_names.index(p) for p in prots]
    k_dapi = next(i for i, n in enumerate(cx_names) if re.match(r'(?i)dapi', n))
    A, field, tile = reg_['A'], reg_['field'], int(reg_['tile'])
    e = expand_um / mpp_he
    m = int(np.ceil(e)) + 2
    n = len(polys)
    bb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    C, HC, WC = z.shape
    AREA = np.zeros(n); AN = np.zeros(n); SUM = np.zeros((n, len(prots))); SD = np.zeros(n); SR = np.zeros(n)
    BAD = np.zeros(n, bool)
    for by in range(0, H, block):
        for bx in range(0, W, block):
            x1, y1 = min(bx + block, W), min(by + block, H)
            sel = np.nonzero((bb[:, 2] >= bx - m) & (bb[:, 0] < x1 + m) & (bb[:, 3] >= by - m) & (bb[:, 1] < y1 + m))[0]
            if not len(sel):
                continue
            lab = np.zeros((y1 - by + 2 * m, x1 - bx + 2 * m), np.int32)
            for j, c in enumerate(sel):
                cv2.fillPoly(lab, [np.round(polys[c] - (bx - m, by - m)).astype(np.int32)], int(j + 1))
            reg_l = expand_labels(lab, distance=e)[m:-m, m:-m]; lab = lab[m:-m, m:-m]
            yy, xx = np.mgrid[by:y1, bx:x1].astype(np.float32)
            ax = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]; ay = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
            dx, dy = field_at(field, tile, ax, ay)
            mx, my = ax + dx, ay + dy                                   # multiplex px of each H&E px
            X0, Y0 = int(max(0, np.floor(mx.min()) - 2)), int(max(0, np.floor(my.min()) - 2))
            X1, Y1 = int(min(WC, np.ceil(mx.max()) + 3)), int(min(HC, np.ceil(my.max()) + 3))
            inside = (mx >= 0) & (my >= 0) & (mx <= WC - 1) & (my <= HC - 1)
            if X1 <= X0 or Y1 <= Y0:
                BAD[sel] = True
                continue
            cod = np.asarray(z[:, Y0:Y1, X0:X1])
            u, v = (mx - X0).astype(np.float32), (my - Y0).astype(np.float32)
            l = reg_l.ravel(); ln = lab.ravel(); nb = len(sel) + 1
            dap = cv2.remap(cod[k_dapi].astype(np.float32), u, v, cv2.INTER_LINEAR, borderValue=0)
            ok_px = inside & (dap > 0)                                  # outside the field / unimaged tile
            bad = np.bincount(l, weights=(~ok_px).ravel().astype(np.float64), minlength=nb)[1:] > 0
            AREA[sel] += np.bincount(l, minlength=nb)[1:]; AN[sel] += np.bincount(ln, minlength=nb)[1:]
            SD[sel] += np.bincount(ln, weights=dap.ravel(), minlength=nb)[1:]
            SR[sel] += np.bincount(l, weights=dap.ravel(), minlength=nb)[1:]
            for jj, ci in enumerate(idx):
                ch = cv2.remap(cod[ci].astype(np.float32), u, v, cv2.INTER_LINEAR, borderValue=0)
                SUM[sel, jj] += np.bincount(l, weights=ch.ravel(), minlength=nb)[1:]
            BAD[sel] |= bad
        log(f'rows {by}/{H}')
    mean = (SUM / np.maximum(AREA, 1)[:, None]).astype(np.float32)
    mean[(AREA == 0) | BAD] = np.nan
    ring = (SR - SD) / np.maximum(AREA - AN, 1); nucm = SD / np.maximum(AN, 1)
    q = (AN > 0) & (AREA - AN > 0) & ~BAD
    qc = float(np.mean(nucm[q] > ring[q])) if q.any() else float('nan')
    return mean, AREA, AN, BAD, qc
