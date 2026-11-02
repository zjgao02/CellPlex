"""Scores of CellPlex.

Per test sample and protein, over all of the sample's cells: Pearson (PCC) and Spearman correlation between
predicted and measured levels, and a cell-level structural similarity (SSIM: prediction and truth each min-max
scaled to [0, 1] over the sample's cells, then SSIM = (2 mu_x mu_y + C1)(2 cov_xy + C2) /
((mu_x^2 + mu_y^2 + C1)(var_x + var_y + C2)), C1 = 0.01^2, C2 = 0.03^2, no spatial window). Proteins whose truth is
constant on a sample, or samples with fewer than 10 cells, are skipped. Cohort score = mean over test samples of the
sample's mean over proteins. Proteins of different panels are matched by canonical name (canon).
"""
import re

import numpy as np
import pandas as pd

C1, C2 = 0.01 ** 2, 0.03 ** 2
ALIAS = {'cytokeratin': 'Pan-CK', 'panck': 'Pan-CK', 'pancytokeratin': 'Pan-CK', 'sma': 'SMA', 'asma': 'SMA',
         'cd3': 'CD3', 'cd3e': 'CD3', 'muc1ema': 'MUC1', 'ccr4cd194': 'CCR4', 'cd194': 'CCR4', 'cd134ox40': 'OX40',
         'betacatenin': 'beta-catenin', 'bcatenin': 'beta-catenin', 'pdpn': 'Podoplanin', 'podoplanin': 'Podoplanin',
         'mct': 'Tryptase', 'tryptase': 'Tryptase', 'pdl1': 'PD-L1', 'pd1': 'PD-1', 'hladr': 'HLA-DR',
         'ecadherin': 'E-cadherin', 'granzymeb': 'Granzyme B', 'collageniv': 'Collagen IV', 'nakatpase': 'Na/K-ATPase',
         'ki67': 'Ki67', 'vimentin': 'Vimentin', 'vim': 'Vimentin', 'tbet': 'T-bet', 'cd279': 'PD-1', 'mhcii': 'HLA-DR',
         'tenascinc': 'TNC', 'periostin': 'POSTN', 'dcxv2': 'DCX'}


def canon(p):
    """canonical protein name of a panel marker: 'CD117 (c-KIT)' -> 'CD117', 'aSMA' -> 'SMA'"""
    k = re.sub(r'[^a-z0-9]', '', re.sub(r'\s*\(.*\)', '', str(p)).lower())
    return ALIAS.get(k, k.upper())


def pcc(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0


def minmax(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def ssim(x, y):
    x, y = minmax(x), minmax(y)
    mx, my = x.mean(), y.mean()
    vx, vy, cxy = x.var(), y.var(), ((x - mx) * (y - my)).mean()
    return float((2 * mx * my + C1) * (2 * cxy + C2) / ((mx * mx + my * my + C1) * (vx + vy + C2)))


def score(res, proteins, with_ssim=False):
    """res: {sample: (cell_idx, pred (n, P), truth (n, P))} -> one row per sample and protein:
    sample, protein, pcc, spearman[, ssim], n_cells"""
    from scipy.stats import spearmanr
    rows = []
    for s, (_, p, t) in res.items():
        for j, g in enumerate(proteins):
            x, y = np.asarray(p[:, j], np.float64), np.asarray(t[:, j], np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if len(x) < 10 or y.std() == 0:
                continue
            r = {'sample': s, 'protein': g, 'pcc': pcc(x, y),
                 'spearman': float(spearmanr(x, y).statistic) if x.std() > 0 else 0.0}
            if with_ssim:
                r['ssim'] = ssim(x, y)
            rows.append(r | {'n_cells': len(x)})
    return pd.DataFrame(rows)


def cohort_score(S, metrics=('pcc', 'spearman')):
    """per (sample, protein) scores -> {metric: mean over samples of the sample's mean over proteins}, n samples"""
    ps = S.groupby('sample')[list(metrics)].mean()
    return {m: float(ps[m].mean()) for m in metrics} | {'n_samples': len(ps)}


def summary(S, metrics=('pcc', 'spearman')):
    """cohort scores and their standard errors over samples"""
    ps = S.groupby('sample')[list(metrics)].mean()
    n = len(ps)
    out = {f'test_{m}': float(ps[m].mean()) for m in metrics}
    out |= {f'test_{m}_sem': float(ps[m].std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan') for m in metrics}
    return out | {'n_test_samples': n}
