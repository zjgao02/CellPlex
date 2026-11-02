"""Linear probe of a frozen pathology foundation model for per-cell protein prediction.

Every 224 px tile (native resolution) goes through the frozen encoder (bf16 autocast); the final patch-token map is
bilinearly sampled at every nucleus centroid (as CELLO's grid sampling, align_corners=True) and a ridge regression maps
the token to the log1p levels of all proteins of the panel.

The encoder is frozen, so a cell's token is the same in every fold: tokens are extracted once per cohort and encoder,
keeping per sample the sufficient statistics (n, sum x, sum y, X'X, X'Y; float64) and, for samples that are validation
or test in an evaluated fold, the tokens of all its cells (fp16). Per fold the ridge is solved on the summed statistics
of the training units (centred, i.e. with intercept), lambda = alpha x the mean eigenvalue of the centred X'X with
alpha on a log grid, chosen by the mean over validation samples of the sample's mean protein PCC (the checkpoint
criterion of the CELLO runs; all cells of the validation samples), and the test units are predicted.

  python scripts/linear_probe.py extract --cohort multitumor --encoder hoptimus0 [--part 0 --nparts 2]   (GPU)
  python scripts/linear_probe.py fit --cohort multitumor --encoder hoptimus0                             (CPU)
Output: token cache $CELLPLEX_DATA/{cohort}/linprobe/{encoder}/{stats,feats}/{sample}.np[yz];
$CELLPLEX_RUNS/{cohort}/linprobe_{encoder}/fold{k}/: probe.npz (W (D, P), b (P), alpha, lam, proteins, encoder),
config.json, test/ (cellplex/engine.py format).
"""
import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cello_utils import slide_gene_pcc  # noqa: E402
from cellplex.cohorts import COHORTS, TILE  # noqa: E402
from cellplex.data import Reader, fold_units, load_meta, load_tiles  # noqa: E402
from cellplex.engine import save_test  # noqa: E402
from cellplex.paths import cohort_dir, run_dir  # noqa: E402

ALPHAS = 10.0 ** np.arange(-6, 1.01, 0.5)


def cache_dir(cohort, enc):
    return f'{cohort_dir(cohort)}/linprobe/{enc}'


def eval_samples(cohort, T):
    """samples that are validation or test in an evaluated fold: their tokens are kept"""
    units = set()
    for k in COHORTS[cohort]['folds']:
        U = fold_units(cohort, k); units |= U['val'] | U['test']
    return set(T['sample'][T.unit.isin(units)])


class Tiles(Dataset):
    def __init__(self, cohort, T):
        self.reader, self.T = Reader(load_meta(cohort)), T.reset_index(drop=True)

    def __len__(self):
        return len(self.T)

    def __getitem__(self, i):
        r = self.T.iloc[i]
        return torch.from_numpy(self.reader.read(r['sample'], r.x0, r.y0)).permute(2, 0, 1), i


def extract(cohort, enc, part, nparts, bs, workers):
    from timm.data import resolve_data_config
    from cellplex.model import ENCODER_REGISTRY, build_encoder
    od = cache_dir(cohort, enc)
    os.makedirs(f'{od}/stats', exist_ok=True); os.makedirs(f'{od}/feats', exist_ok=True)
    T = load_tiles(cohort)
    keep = eval_samples(cohort, T)
    samples = sorted(T['sample'].unique())[part::nparts]
    todo = [s for s in samples if not os.path.exists(f'{od}/stats/{s}.npz')]
    print(f'{cohort} {enc} part {part}/{nparts}: {len(samples)} samples, {len(todo)} to do, {len(keep)} keep tokens', flush=True)
    if not todo:
        return
    dev = torch.device('cuda')
    m = build_encoder(enc, pretrained=True).to(dev).eval()
    cfg = resolve_data_config(m.pretrained_cfg, model=m)
    mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)
    npre, D = getattr(m, 'num_prefix_tokens', 1), m.embed_dim
    g = TILE // ENCODER_REGISTRY[enc]['patch_size']
    for s in todo:
        t0 = time.time()
        Ts = T[T['sample'] == s].sort_values('c0').reset_index(drop=True)
        C = np.load(f'{cohort_dir(cohort)}/cells/{s}.npz')
        cx, cy, Y = C['cx'], C['cy'], torch.from_numpy(C['y'].astype(np.float64)).to(dev)
        N, P = Y.shape
        X = torch.empty((N, D), dtype=torch.float16, device=dev)
        dl = DataLoader(Tiles(cohort, Ts), batch_size=bs, num_workers=workers, pin_memory=True)
        with torch.no_grad():
            for img, idx in dl:
                x = (img.to(dev, non_blocking=True).float() / 255. - mean) / (std + 1e-6)   # CELLO's normalisation
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tok = m.forward_features(x)[:, npre:, :]
                fm = tok.float().view(len(idx), g, g, D).permute(0, 3, 1, 2)             # row-major token grid
                r = Ts.iloc[idx.numpy()]
                n = (r.c1 - r.c0).values; nmax = int(n.max())
                grid = torch.zeros((len(idx), 1, max(nmax, 1), 2), device=dev)
                for b, (c0, c1, x0, y0) in enumerate(zip(r.c0, r.c1, r.x0, r.y0)):
                    xy = torch.from_numpy(np.stack([cx[c0:c1] - x0, cy[c0:c1] - y0], 1)).to(dev)
                    grid[b, 0, :c1 - c0] = xy / TILE * 2 - 1
                q = F.grid_sample(fm, grid, mode='bilinear', align_corners=True)[:, :, 0, :]   # (B, D, nmax)
                for b, (c0, c1) in enumerate(zip(r.c0, r.c1)):
                    X[c0:c1] = q[b, :, :c1 - c0].T.half()
        Xd = X.double()
        st = dict(n=N, sx=Xd.sum(0).cpu().numpy(), sy=Y.sum(0).cpu().numpy(), xx=(Xd.T @ Xd).cpu().numpy(),
                  xy=(Xd.T @ Y).cpu().numpy())
        if s in keep:
            np.save(f'{od}/feats/{s}.npy', X.cpu().numpy())
        tmp = f'{od}/stats/{s}.{os.getpid()}.tmp.npz'
        np.savez(tmp, **st); os.replace(tmp, f'{od}/stats/{s}.npz')
        dt = time.time() - t0
        print(f'{s}: {len(Ts):,} tiles, {N:,} cells, {dt:.0f}s ({len(Ts) / dt:.0f} tiles/s){" tokens kept" if s in keep else ""}',
              flush=True)
        del X, Xd


def fit(cohort, enc, folds=None):
    od, T = cache_dir(cohort, enc), load_tiles(cohort)
    proteins = list(load_meta(cohort)['proteins'])
    by_unit = T.groupby('unit')['sample'].unique()
    st = {s: dict(np.load(f'{od}/stats/{s}.npz')) for s in T['sample'].unique()}
    truth = lambda s: np.load(f'{cohort_dir(cohort)}/cells/{s}.npz')['y'].astype(np.float32)   # noqa: E731
    feats = lambda s: np.load(f'{od}/feats/{s}.npy').astype(np.float32)                      # noqa: E731
    for k in folds if folds is not None else COHORTS[cohort]['folds']:
        U = fold_units(cohort, k)
        smp = {key: sorted({s for u in U[key] for s in by_unit.get(u, [])}) for key in U}
        n = sum(st[s]['n'] for s in smp['train'])
        sx = sum(st[s]['sx'] for s in smp['train']); sy = sum(st[s]['sy'] for s in smp['train'])
        mx, my = sx / n, sy / n
        Cxx = sum(st[s]['xx'] for s in smp['train']) - n * np.outer(mx, mx)
        Cxy = sum(st[s]['xy'] for s in smp['train']) - n * np.outer(mx, my)
        e, V = np.linalg.eigh(Cxx); e = np.clip(e, 0, None)
        VB = V.T @ Cxy
        lams = ALPHAS * e.mean()
        Ws = [(V @ (VB / (e + l)[:, None])).astype(np.float32) for l in lams]
        bs = [(my - mx @ W).astype(np.float32) for W in Ws]
        val = np.zeros(len(lams))
        for s in smp['val']:
            Xv, Yv = feats(s), truth(s)
            val += [slide_gene_pcc(Xv @ W + b, Yv) for W, b in zip(Ws, bs)]
        val /= max(len(smp['val']), 1)
        j = int(np.argmax(val))
        out = run_dir(cohort, f'linprobe_{enc}', k)
        os.makedirs(out, exist_ok=True)
        np.savez(f'{out}/probe.npz', W=Ws[j], b=bs[j], alpha=ALPHAS[j], lam=lams[j], proteins=np.array(proteins),
                 encoder=enc, feat_dim=Ws[j].shape[0])
        res = {}
        for s in smp['test']:
            p, t = feats(s) @ Ws[j] + bs[j], truth(s)
            res[s] = (np.arange(len(t)), p, t)
        _, summ = save_test(out, res, proteins, {'val_pcc': float(val[j]), 'alpha': float(ALPHAS[j]),
                                                 'lambda': float(lams[j]), 'n_train_cells': int(n)})
        json.dump({'method': 'linear probe (frozen encoder, centroid token, ridge)', 'encoder': enc, 'cohort': cohort,
                   'fold': k, 'val_curve': dict(zip(map(float, ALPHAS), map(float, val))),
                   'units': {key: sorted(v) for key, v in U.items()}}, open(f'{out}/config.json', 'w'), indent=1)
        print(f'{cohort} {enc} fold {k}: {n:,} train cells, alpha {ALPHAS[j]:.0e} (val {val[j]:.4f}), '
              f'TEST PCC {summ["test_pcc"]:.4f} +- {summ["test_pcc_sem"]:.4f} over {summ["n_test_samples"]} samples', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'fit'])
    ap.add_argument('--cohort', required=True, help='comma-separated')
    ap.add_argument('--encoder', required=True, help='uni2 | virchow2 | hoptimus0')
    ap.add_argument('--folds', default='', help='fit: comma-separated folds (default: the evaluated folds of the cohort)')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=8)
    a = ap.parse_args()
    for c in a.cohort.split(','):
        if a.cmd == 'extract':
            extract(c, a.encoder, a.part, a.nparts, a.batch_size, a.num_workers)
        else:
            fit(c, a.encoder, [int(k) for k in a.folds.split(',')] if a.folds else None)


if __name__ == '__main__':
    main()
