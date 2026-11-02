"""PCC and SSIM tables of every method from the test predictions of its runs (cellplex/metrics.py).

For each cohort and method, the test outputs of every evaluated fold ($CELLPLEX_RUNS/{cohort}/{method}/fold{k}/test/
pred_{sample}.npz: the out-of-fold predictions of five-fold cohorts, the fixed test set of Orion-CRC and glioma) are
scored per sample and protein. Cohort score = mean over test samples of the sample's mean over proteins (+- standard
error over samples); per-protein score = mean over test samples.

  python scripts/evaluate.py                                   every cohort and method with test outputs
  python scripts/evaluate.py --cohorts multitumor --methods hoptimus0,eva_ft
Output ($CELLPLEX_RUNS/results/ or --out): scores_{cohort}.csv (method, sample, protein, pcc, spearman, ssim, n_cells),
cohort_scores.csv, proteins_{cohort}.csv (per-protein PCC and SSIM of every method), results.md.
"""
import argparse, glob, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cohorts import COHORTS  # noqa: E402
from cellplex.metrics import score  # noqa: E402
from cellplex.paths import RUNS, run_dir  # noqa: E402

METHODS = {'eva_zeroshot': 'Eva, zero-shot', 'eva_ft': 'Eva, fine-tuned', 'gigatime_zeroshot': 'GigaTIME, zero-shot',
           'gigatime_ft': 'GigaTIME, fine-tuned', 'linprobe_uni2': 'Linear probe, UNI2-h',
           'linprobe_virchow2': 'Linear probe, Virchow2', 'linprobe_hoptimus0': 'Linear probe, H-optimus-0',
           'uni2': 'CELLO, UNI2-h', 'virchow2': 'CELLO, Virchow2', 'hoptimus0': 'CELLO, H-optimus-0'}
METRICS = ('pcc', 'ssim')


def load(cohort, method):
    """{sample: (cell_idx, pred, truth)} and the protein names of a method's test outputs, or (None, None)"""
    res, proteins = {}, None
    for k in COHORTS[cohort]['folds']:
        fs = sorted(glob.glob(f'{run_dir(cohort, method, k)}/test/pred_*.npz'))
        if not fs:
            return None, None
        for f in fs:
            P = np.load(f, allow_pickle=True)
            proteins = [str(p) for p in P['proteins']]
            res[os.path.basename(f)[5:-4]] = (P['cell_idx'], P['pred'].astype(np.float64), P['truth'].astype(np.float64))
    return res, proteins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohorts', default=','.join(COHORTS))
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--out', default=f'{RUNS}/results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, L = [], ['# CellPlex results', '', 'Mean +- standard error over test samples of the sample\'s mean over proteins.', '']
    for c in a.cohorts.split(','):
        R = []
        for m in a.methods.split(','):
            res, proteins = load(c, m)
            if res is None:
                continue
            S = score(res, proteins, with_ssim=True).assign(method=m)
            R.append(S)
            ps = S.groupby('sample')[list(METRICS)].mean()
            r = {'cohort': c, 'method': m, 'n_samples': len(ps), 'n_proteins': len(proteins)}
            for k in METRICS:
                r[k], r[f'{k}_sem'] = float(ps[k].mean()), float(ps[k].std(ddof=1) / np.sqrt(len(ps)))
            rows.append(r)
            print(f'{COHORTS[c]["name"]:12s} {METHODS.get(m, m):28s} PCC {r["pcc"]:.3f} +- {r["pcc_sem"]:.3f}  '
                  f'SSIM {r["ssim"]:.3f} +- {r["ssim_sem"]:.3f}  ({len(ps)} samples, {len(proteins)} proteins)', flush=True)
        if not R:
            continue
        R = pd.concat(R)
        R[['method', 'sample', 'protein', 'pcc', 'spearman', 'ssim', 'n_cells']].to_csv(f'{a.out}/scores_{c}.csv', index=False)
        Pt = R.groupby(['protein', 'method'])[list(METRICS)].mean().unstack('method')
        Pt.columns = [f'{m}_{k}' for k, m in Pt.columns]
        Pt.reset_index().to_csv(f'{a.out}/proteins_{c}.csv', index=False)
    T = pd.DataFrame(rows)
    T.to_csv(f'{a.out}/cohort_scores.csv', index=False)
    for c, g in T.groupby('cohort', sort=False):
        L += [f'## {COHORTS[c]["name"]}', '', '| method | PCC | SSIM | samples | proteins |', '|---|---|---|---|---|']
        L += [f'| {METHODS.get(r.method, r.method)} | {r.pcc:.3f} +- {r.pcc_sem:.3f} | {r.ssim:.3f} +- {r.ssim_sem:.3f} | '
              f'{r.n_samples} | {r.n_proteins} |' for r in g.itertuples()]
        L.append('')
    open(f'{a.out}/results.md', 'w').write('\n'.join(L) + '\n')
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
