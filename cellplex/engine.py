"""Prediction over the cells of a split and the test outputs shared by every method.

Test outputs of a run directory ({run}/test/): pred_{sample}.npz (cell_idx = rows of cells/{sample}.npz, pred and truth
(n, P) float32, proteins), per_sample_protein.csv (sample, protein, pcc, spearman, n_cells) and summary.json.
"""
import json, os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cello_utils import collate_fn, is_foreground_tile
from .metrics import score, summary


class FlagDS(Dataset):
    """Adds CELLO's whole-slide foreground flag (validation only)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        it['keep'] = bool(is_foreground_tile((it['img'].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)))
        return it


def collate_keep(b):
    b = [x for x in b if x.get('keep', True)]
    if not b:
        return None
    out = collate_fn(b)
    out['cell_idx'] = [x['cell_idx'] for x in b]
    return out


@torch.no_grad()
def predict(model, ds, device, bs, workers, foreground=False, amp=False):
    """-> {sample: (cell_idx, pred, truth)} over every cell of every (foreground) tile of a TileDS."""
    pmap = {g: i for i, g in enumerate(model.union_genes)}
    dl = DataLoader(FlagDS(ds) if foreground else ds, batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=collate_keep)
    acc = {}
    model.eval()
    for batch in dl:
        if batch is None:
            continue
        with torch.amp.autocast('cuda', enabled=amp):
            out = model.forward_batch(batch, pmap, compute_preds=True)
        if out['union_logits'] is None:
            continue
        p = out['val_cell_pred'].float().cpu().numpy()
        t = out['val_cell_gt'].float().cpu().numpy()
        s0 = 0
        for k, (pat, n) in enumerate(zip(batch['patient'], batch['n_cells'])):
            idx = batch['cell_idx'][k].numpy()
            a = acc.setdefault(pat, ([], [], []))
            a[0].append(idx); a[1].append(p[s0:s0 + n]); a[2].append(t[s0:s0 + n])
            s0 += n
    return {s: tuple(np.concatenate(x) for x in v) for s, v in acc.items()}


def save_test(out, res, proteins, extra=None):
    """write the test outputs of res {sample: (cell_idx, pred, truth)} to {out}/test; returns (scores, summary)"""
    os.makedirs(f'{out}/test', exist_ok=True)
    for s, (idx, p, t) in res.items():
        np.savez(f'{out}/test/pred_{s}.npz', cell_idx=idx, pred=np.asarray(p, np.float32),
                 truth=np.asarray(t, np.float32), proteins=np.array(proteins))
    S = score(res, proteins)
    S.to_csv(f'{out}/test/per_sample_protein.csv', index=False)
    summ = summary(S) | dict(extra or {})
    json.dump(summ, open(f'{out}/test/summary.json', 'w'), indent=1)
    return S, summ


class _NoLog:
    summary = {}

    def log(self, *a, **k):
        pass

    def finish(self):
        pass


def wandb_run(mode, **kw):
    """a Weights & Biases run (resumed by id), or a no-op logger when mode == 'disabled'"""
    if mode == 'disabled':
        return _NoLog()
    import wandb
    return wandb.init(mode=mode, name=kw.get('id'), resume='allow', **kw)
