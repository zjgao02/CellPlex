"""Steps shared by the cohort scripts of scripts/preprocess/.

Every cohort follows one protocol: CellViT++ nuclei segmented on the H&E at 0.25 um/px (its native network
resolution; the H&E is resampled to it when the scan is coarser) and mapped back to the H&E's native pixels; cell
region = nucleus expanded by 2 um without entering neighbours (skimage expand_labels); truth = mean multiplex
intensity in the region, after moving the regions into the multiplex frame (an integer shift for the array cohorts,
a registration for the whole slides). QC per sample: the fraction of cells whose nuclear stain (Hoechst / DAPI) is
brighter in the nucleus than in the ring around it; samples below MIN_QC are excluded by build_dataset.py.

Resampling convention (one global pixel-centre rule, so coordinates map back exactly):
    x_in = (x_out + 0.5) / sx - 0.5,   sx = W_out / W_in   (same for y)
"""
import json

import cv2
import numpy as np
import tifffile

CV_MPP = 0.25                    # CellViT++ input resolution (SAM-H x40)
EXPAND_UM = 2.0                  # cell region = nucleus + 2 um
MIN_QC = 0.80                    # nucleus brighter than its ring in >= 80% of cells


def lin(n_out, s, n_in):
    """Source coordinate of output pixels 0..n_out-1 (clipped to the image)."""
    return np.clip((np.arange(n_out, dtype=np.float32) + 0.5) / s - 0.5, 0, n_in - 1)


def write_pyramid(path, img, mpp_):
    """(H, W, 3) uint8 -> pyramidal JPEG TIFF (256 px tiles, 2x levels down to ~1024 px), readable by OpenSlide."""
    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        opts = dict(tile=(256, 256), compression='jpeg', compressionargs={'level': 90}, photometric='rgb',
                    resolutionunit='CENTIMETER')
        lv, k = img, 0
        tw.write(lv, resolution=(1e4 / mpp_, 1e4 / mpp_), subifds=0, **opts)
        while min(lv.shape[:2]) > 1024:
            k += 1
            lv = cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA)
            tw.write(lv, subfiletype=1, resolution=(1e4 / (mpp_ * 2 ** k), 1e4 / (mpp_ * 2 ** k)), **opts)


def cellvit_input(he, mpp, tif, js, **extra):
    """Write the H&E (native mpp) resampled bilinearly to CV_MPP as a pyramidal TIFF + its geometry json."""
    H, W = he.shape[:2]
    Ho, Wo = int(round(H * mpp / CV_MPP)), int(round(W * mpp / CV_MPP))
    sx, sy = Wo / W, Ho / H
    mx, my = np.meshgrid(lin(Wo, sx, W), lin(Ho, sy, H))
    write_pyramid(tif, cv2.remap(he, mx, my, interpolation=cv2.INTER_LINEAR), CV_MPP)
    json.dump({**extra, 'W': W, 'H': H, 'Wo': Wo, 'Ho': Ho, 'sx': sx, 'sy': sy, 'mpp': mpp, 'mpp_cv': CV_MPP,
               'map_back': 'x_in = (x + 0.5) / sx - 0.5'}, open(js, 'w'), indent=1)
    return Ho, Wo


def read_cellvit(path, sx=1.0, sy=1.0):
    """CellViT++ {name}_cells.json -> (contours, centroids (N, 2), classes) in native px of the H&E it was run on
    (sx, sy: the resampling factors of cellvit_input; 1 when CellViT ran at native resolution)."""
    J = json.load(open(path))
    md = J['wsi_metadata']
    assert abs(md['base_mpp'] - md['target_patch_mpp']) < 0.035, f'CellViT rescaled internally: {md}'
    tmap = {int(k): v for k, v in J['type_map'].items()}
    native = sx == 1 and sy == 1
    polys, cen, typ = [], [], []
    for cell in J['cells']:
        r, c = np.asarray(cell['contour'], np.float64), np.asarray(cell['centroid'], np.float64)
        polys.append(r if native else (r + 0.5) / (sx, sy) - 0.5)
        cen.append(c if native else (c + 0.5) / (sx, sy) - 0.5)
        typ.append(tmap[cell['type']])
    return polys, np.asarray(cen).reshape(-1, 2), typ


def label_image(polys, H, W):
    """Nuclei 1..N drawn on an (H, W) int32 image, in order (later nuclei overwrite earlier ones)."""
    lab = np.zeros((H, W), np.int32)
    for k, r in enumerate(polys):
        cv2.fillPoly(lab, [np.round(r).astype(np.int32)], k + 1)
    return lab


def estimate_shift(nuc, hoechst, search=40):
    """Integer (dy, dx) with multiplex = H&E + (dy, dx): peak of the FFT cross-correlation of the nucleus mask (H&E
    frame) with the nuclear stain (multiplex frame) within +-search px."""
    m = cv2.GaussianBlur((nuc > 0).astype(np.float32), (0, 0), 1.0)
    h = hoechst.astype(np.float32)
    h = np.clip(h / max(np.percentile(h, 99.5), 1e-6), 0, 1)
    h = cv2.GaussianBlur(h, (0, 0), 1.0)
    m, h = m - m.mean(), h - h.mean()
    C = np.fft.ifft2(np.fft.fft2(h) * np.conj(np.fft.fft2(m))).real          # C(d) = sum_x h(x + d) m(x)
    C = np.fft.fftshift(C)
    cy, cx = C.shape[0] // 2, C.shape[1] // 2
    Wn = C[cy - search:cy + search + 1, cx - search:cx + search + 1]
    iy, ix = np.unravel_index(np.argmax(Wn), Wn.shape)
    return int(iy - search), int(ix - search)


def shift_labels(inst, dy, dx):
    """Move a label image from the H&E frame to the multiplex frame (pixel (y, x) -> (y + dy, x + dx))."""
    out = np.zeros_like(inst)
    H, W = inst.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = inst[ys, xs]
    return out


def nuc_vs_ring(lab, inst, hv, n):
    """Fraction of cells whose nuclear stain is brighter in the nucleus than in the 2 um ring around it."""
    a_n = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    a_c = np.bincount(inst.ravel(), minlength=n + 1)[1:]
    s_n = np.bincount(lab.ravel(), weights=hv, minlength=n + 1)[1:]
    s_c = np.bincount(inst.ravel(), weights=hv, minlength=n + 1)[1:]
    q = (a_n > 0) & (a_c - a_n > 0)
    return float(np.mean((s_n[q] / a_n[q]) > (s_c[q] - s_n[q]) / (a_c[q] - a_n[q]))) if q.any() else float('nan')


def quant_shifted(polys, cen, typ, H, W, mpp, cod, hidx, pidx):
    """Array cohorts: regions in the H&E frame, the per-sample shift that best overlays the nuclei on the nuclear
    stain, truth = mean of every channel of pidx ({name: index}) in the shifted regions -> dict for quant/{s}.npz."""
    from skimage.segmentation import expand_labels
    n = len(cen)
    lab = label_image(polys, H, W)
    nuc_area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    inst = expand_labels(lab, distance=EXPAND_UM / mpp)                  # H&E frame (predictions are aggregated here)
    hz = np.asarray(cod[hidx])
    hv = hz.ravel().astype(np.float64)
    qc0 = nuc_vs_ring(lab, inst, hv, n)
    dy, dx = estimate_shift(lab, hz)
    lab_c, inst_c = shift_labels(lab, dy, dx), shift_labels(inst, dy, dx)   # multiplex frame (truth is read here)
    qc1 = nuc_vs_ring(lab_c, inst_c, hv, n)
    area = np.bincount(inst_c.ravel(), minlength=n + 1)[1:]
    l = inst_c.ravel()
    ok = area > 0
    keys = list(pidx)
    mean = np.full((n, len(keys)), np.nan, np.float32)
    for j, p in enumerate(keys):
        s = np.bincount(l, weights=np.asarray(cod[pidx[p]]).ravel().astype(np.float64), minlength=n + 1)[1:]
        mean[ok, j] = (s[ok] / area[ok]).astype(np.float32)
    return dict(keys_all=np.array(keys), mean_all=mean, cx=cen[:, 0].astype(np.float32), cy=cen[:, 1].astype(np.float32),
                cls=np.array(typ), area_nuc=nuc_area.astype(np.int32), area=area.astype(np.int32),
                inst=inst.astype(np.int32), mpp=mpp, expand_um=EXPAND_UM, shift=np.array([dy, dx]),
                hoechst_nuc_gt_ring=qc1, hoechst_nuc_gt_ring_unshifted=qc0)


def tissue_of(he):
    """H&E tissue mask: saturation > 20 after a 5 px blur (background is ~5)."""
    return cv2.blur(cv2.cvtColor(np.ascontiguousarray(he), cv2.COLOR_RGB2HSV)[..., 1], (5, 5)) > 20


def write_filelist(path, rows):
    """CellViT++ process_dataset file list: (image path, slide mpp, magnification) per row."""
    open(path, 'w').write('path,slide_mpp,magnification\n' + ''.join(f'{p},{m},{g}\n' for p, m, g in rows))
