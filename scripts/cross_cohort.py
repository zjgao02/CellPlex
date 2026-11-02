"""Transfer across cohorts: the fold-0 CELLO model of one cohort predicts the cells of every other cohort without further
training, scored on the proteins the two panels share (matched by canonical name, cellplex.metrics.canon).

The target H&E is resampled to the source's training resolution: the target cells are re-tiled on a grid of
R = 224 * mpp_src / mpp_tgt target px, each tile is read at the target's native resolution and resized to 224 px (area
when shrinking, bilinear when enlarging), and the cell centroids are scaled with it. Evaluation cells = the test units
of every evaluated fold of the target (the cells of its in-domain scores).
Summary: cross = the source's model; in_domain = the target's own models with the same encoder (out-of-fold), on the
same samples and proteins; retention = cross / in_domain; score = mean over samples of the sample's mean protein PCC.

  python scripts/cross_cohort.py run --src crc_tma --encoder hoptimus0                 every other cohort
  python scripts/cross_cohort.py run --src orion_crc --encoder hoptimus0 --targets bc_crc
  python scripts/cross_cohort.py summary                                               -> summary.csv, cross_cohort.md
Output $CELLPLEX_RUNS/cross_cohort/{src}/{encoder}/{target}/: pred_{sample}.npz, per_sample_protein.csv, summary.json
(a target with summary.json is skipped).
"""
import argparse, glob, json, os, sys, time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cohorts import COHORTS, TILE  # noqa: E402
from cellplex.data import Reader, load_meta, load_tiles  # noqa: E402
from cellplex.metrics import canon, score, summary  # noqa: E402
from cellplex.model import load_checkpoint  # noqa: E402
from cellplex.paths import RUNS, cohort_dir, run_dir  # noqa: E402

OUT = f'{RUNS}/cross_cohort'
ENC = {'uni2': 'UNI2-h', 'virchow2': 'Virchow2', 'hoptimus0': 'H-optimus-0'}


def eval_units(cohort):
    F = pd.read_csv(f'{cohort_dir(cohort)}/folds.csv', dtype={'unit': str})
    return set(F.unit[F.fold.isin(COHORTS[cohort]['folds'])])


def shared(src_prots, tgt_prots):
    """[(target column, source column)] of the proteins both panels measure (first occurrence of a canonical name)"""
    s = {}
    for j, p in enumerate(src_prots):
        s.setdefault(canon(p), j)
    out, seen = [], set()
    for j, p in enumerate(tgt_prots):
        c = canon(p)
        if c in s and c not in seen:
            out.append((j, s[c])); seen.add(c)
    return out


class Regrid(Dataset):
    """the target's evaluation cells on a grid of R = TILE * mpp_src / mpp_tgt target px, served at TILE px"""

    def __init__(self, tgt, mpp_src):
        self.meta = load_meta(tgt)
        self.reader = Reader(self.meta)
        R = TILE * mpp_src / self.meta['mpp']
        self.n = int(round(R))
        T = load_tiles(tgt)
        T = T[T.unit.isin(eval_units(tgt))]
        self.idx, self.lxy, self.items = {}, {}, []
        for s, g in T.groupby('sample', sort=True):
            C = np.load(f'{cohort_dir(tgt)}/cells/{s}.npz')
            idx = np.concatenate([np.arange(a, b) for a, b in zip(g.c0.values, g.c1.values)])
            cx, cy = C['cx'][idx].astype(np.float64), C['cy'][idx].astype(np.float64)
            gx, gy = np.floor(cx / R).astype(np.int64), np.floor(cy / R).astype(np.int64)
            o = np.lexsort((gx, gy))
            idx, cx, cy, gx, gy = idx[o], cx[o], cy[o], gx[o], gy[o]
            key = gy * 10_000_000 + gx
            starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
            ends = np.r_[starts[1:], len(key)]
            x0 = np.round(gx[starts] * R).astype(np.int64); y0 = np.round(gy[starts] * R).astype(np.int64)
            lx = cx - np.repeat(x0, ends - starts); ly = cy - np.repeat(y0, ends - starts)
            f = TILE / self.n
            self.lxy[s] = np.clip(np.stack([lx * f, ly * f], 1), 0, TILE - 1e-3).astype(np.float32)
            self.idx[s] = idx
            self.items += [(s, int(a), int(b), int(c), int(d)) for a, b, c, d in zip(x0, y0, starts, ends)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, x0, y0, a, b = self.items[i]
        img = self.reader.read(s, x0, y0, T=self.n)
        if self.n != TILE:
            img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA if self.n > TILE else cv2.INTER_LINEAR)
        return (torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.),
                torch.from_numpy(self.lxy[s][a:b]), (s, a, b))


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b], [x[2] for x in b]


def run_target(model, src_prots, mpp_src, tgt, out, bs, workers, log):
    meta = load_meta(tgt)
    tp = list(meta['proteins'])
    pairs = shared(src_prots, tp)
    if not pairs:
        log(f'{tgt}: no shared proteins, skipped'); return
    tcol, scol = [p[0] for p in pairs], [p[1] for p in pairs]
    D = Regrid(tgt, mpp_src)
    n_cells = sum(len(v) for v in D.idx.values())
    log(f'{tgt}: {len(pairs)} shared proteins, {len(D.idx)} samples, {n_cells:,} cells, {len(D):,} tiles of '
        f'{D.n} target px (mpp {meta["mpp"]:.4f} -> {mpp_src:.4f})')
    pred = {s: np.full((len(v), len(pairs)), np.nan, np.float32) for s, v in D.idx.items()}
    dl = DataLoader(D, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate,
                    prefetch_factor=4)
    t0 = last = time.time()
    with torch.no_grad():
        for k, (imgs, xy, where) in enumerate(dl):
            outs = model.predict(imgs.cuda(non_blocking=True), xy)
            for (s, a, b), o in zip(where, outs):
                pred[s][a:b] = o.numpy()[:, scol]
            if time.time() - last > 120:
                last = time.time()
                log(f'  {tgt}: {(k + 1) * bs:,} / {len(D):,} tiles, {(k + 1) * bs / (last - t0):.0f} tiles/s')
    os.makedirs(out, exist_ok=True)
    res = {}
    names = [tp[j] for j in tcol]
    for s, idx in D.idx.items():
        y = np.load(f'{cohort_dir(tgt)}/cells/{s}.npz')['y'][idx][:, tcol].astype(np.float32)
        res[s] = (idx, pred[s], y)
        np.savez(f'{out}/pred_{s}.npz', cell_idx=idx, pred=pred[s], truth=y, proteins=np.array(names))
    S = score(res, names)
    S['protein_c'] = S.protein.map(canon)
    S['src_protein'] = S.protein.map({tp[j]: src_prots[c] for j, c in pairs})
    S.to_csv(f'{out}/per_sample_protein.csv', index=False)
    summ = summary(S) | {'n_cells': n_cells, 'n_shared': len(pairs), 'proteins': names, 'mpp_src': mpp_src,
                         'mpp_tgt': meta['mpp'], 'tile_px_target': D.n, 'seconds': round(time.time() - t0)}
    json.dump(summ, open(f'{out}/summary.json', 'w'), indent=1)
    log(f'{tgt}: PCC {summ["test_pcc"]:.4f} +- {summ["test_pcc_sem"]:.4f} over {summ["n_test_samples"]} samples, '
        f'{len(pairs)} proteins; {summ["seconds"]}s')


def run(a):
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    def log(m):
        print(f'{time.strftime("%H:%M:%S")} {m}', flush=True)
    ck = a.ckpt or f'{run_dir(a.src, a.encoder, a.fold)}/best_model.pt'
    model, cfg, prots = load_checkpoint(ck, device='cuda')
    mpp_src = load_meta(a.src)['mpp']
    log(f'{a.src} {a.encoder} fold {a.fold}: {len(prots)} proteins, mpp {mpp_src:.4f}, {torch.cuda.get_device_name()}')
    for tgt in a.targets.split(',') if a.targets else [c for c in COHORTS if c != a.src]:
        out = f'{OUT}/{a.src}/{a.encoder}/{tgt}'
        if os.path.exists(f'{out}/summary.json'):
            log(f'{tgt}: done before'); continue
        run_target(model, prots, mpp_src, tgt, out, a.batch_size, a.num_workers, log)


def in_domain(cohort, method):
    """per (sample, protein) PCC of a cohort's own runs over its evaluated folds, with canonical names, or None"""
    fs = [f'{run_dir(cohort, method, k)}/test/per_sample_protein.csv' for k in COHORTS[cohort]['folds']]
    if not all(os.path.exists(f) for f in fs):
        return None
    D = pd.concat([pd.read_csv(f, dtype={'sample': str}) for f in fs])
    D['protein_c'] = D.protein.map(canon)
    return D


def mean_score(f, prots, samples):
    g = f[f.protein_c.isin(prots) & f['sample'].isin(samples)]
    return g.groupby('sample').pcc.mean().mean() if len(g) else np.nan


def summarize(a):
    rows = []
    for sj in sorted(glob.glob(f'{OUT}/*/*/*/summary.json')):
        tgt, enc, src = sj.split('/')[-2], sj.split('/')[-3], sj.split('/')[-4]
        S = pd.read_csv(f'{os.path.dirname(sj)}/per_sample_protein.csv', dtype={'sample': str})
        P, smp = set(S.protein_c), set(S['sample'])
        r = dict(src=src, tgt=tgt, encoder=enc, n_shared=len(P), n_samples=len(smp), cross=mean_score(S, P, smp))
        D = in_domain(tgt, enc)
        r['in_domain'] = mean_score(D, P, smp) if D is not None else np.nan
        r['retention'] = r['cross'] / r['in_domain']
        Z = in_domain(tgt, 'eva_zeroshot')                       # Eva as released, on the same samples and proteins
        r['eva_zeroshot'] = mean_score(Z, P, smp) if Z is not None else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/summary.csv', index=False)
    order = [c for c in COHORTS if c in set(R.src) | set(R.tgt)]
    L = ['# Transfer across cohorts (fold-0 CELLO model of the training cohort, H&E resampled to its resolution)', '',
         'Mean over the target\'s samples of the mean PCC over the shared proteins. Rows: training cohort; columns: '
         'target cohort; diagonal: the target\'s own models on all its proteins; off-diagonal: cross (the target\'s own '
         'models on the same proteins and cells).', '']
    for enc in [e for e in ENC if e in set(R.encoder)]:
        E = R[R.encoder == enc]
        L += [f'## {ENC[enc]}', '', '| trained on | ' + ' | '.join(COHORTS[c]['name'] for c in order) + ' |',
              '|---|' + '---|' * len(order)]
        for s in order:
            cells = []
            for t in order:
                if s == t:
                    D = in_domain(t, enc)
                    cells.append(f'**{D.groupby("sample").pcc.mean().mean():.3f}**' if D is not None else '')
                    continue
                q = E[(E.src == s) & (E.tgt == t)]
                cells.append(f'{q.cross.iloc[0]:.3f} ({q.in_domain.iloc[0]:.3f})' if len(q) else '')
            L.append(f'| {COHORTS[s]["name"]} | ' + ' | '.join(cells) + ' |')
        L += ['', f'Mean over {len(E)} pairs: cross {E.cross.mean():.3f}, in-domain {E.in_domain.mean():.3f}, '
                  f'retention {E.retention.mean():.0%}, Eva zero-shot {E.eva_zeroshot.mean():.3f}.', '']
    open(f'{OUT}/cross_cohort.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'summary'])
    ap.add_argument('--src', default='')
    ap.add_argument('--encoder', default='hoptimus0', help='uni2 | virchow2 | hoptimus0')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--ckpt', default='', help='default: $CELLPLEX_RUNS/{src}/{encoder}/fold{fold}/best_model.pt')
    ap.add_argument('--targets', default='', help='comma list (default: every other cohort)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=8)
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else summarize(a)


if __name__ == '__main__':
    main()
