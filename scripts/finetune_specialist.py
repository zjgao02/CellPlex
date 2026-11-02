"""Eva / GigaTIME on one fold of one cohort: fine-tuned on the fold's training units with their own pixel-level
objectives (--zeroshot: released weights, no training), scored like the CELLO runs (same cells, truth and metrics).

  eva       H&E only (1 - RGB/255, HECHA1-3) at 0.5 um/px, 224 windows; every panel protein requested by an Eva marker
            name (cellplex/specialists/eva.py); masked MSE against the [0, 1] pixel target; AdamW lr 1e-4 x
            sqrt(batch / 64), wd 0.05, 300 warm-up steps, cosine, clip 1, batch 32; all parameters trained
  gigatime  ImageNet-normalised RGB at 0.25 um/px, 128 um crops = 512 px as 4 windows of 256; one output per panel
            protein (cellplex/specialists/gigatime.py); BCE + Dice on tissue pixels; AdamW lr 3e-4 x sqrt(batch / 256),
            wd 1e-4, 200 warm-up steps, cosine, clip 1, batch 64 crops
Training length (--epochs): passes over the tissue of the fold's training tiles, as windows of the model's field of
view; the checkpoint is the best of --n_val validations on the fold's
validation units (mean over samples of the mean protein PCC; whole-slide cohorts: a fixed random subset of their
2048 px blocks). Optional early stopping (--patience, --min_delta) and time budget (--max_hours). Test: every cell of
the test units, its value = mean prediction over its nucleus + 2 um region (cellplex/specialists/data.py).
Eva and GigaTIME are installed from their own repositories (docs/specialists.md).

  python scripts/finetune_specialist.py --model eva --cohort multitumor --fold 0 --epochs N
  python scripts/finetune_specialist.py --model gigatime --cohort multitumor --fold 0 --zeroshot
Output: $CELLPLEX_RUNS/{cohort}/{model}_{ft|zeroshot}/fold{k}/ (config.json, train.log, best_model.pt, test/).
"""
import argparse, json, math, os, sys, time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cello_utils import slide_gene_pcc  # noqa: E402
from cellplex.cohorts import TILE  # noqa: E402
from cellplex.data import fold_units, load_meta, load_tiles  # noqa: E402
from cellplex.engine import save_test, wandb_run  # noqa: E402
from cellplex.paths import REPO, run_dir  # noqa: E402
from cellplex.specialists import Specialist  # noqa: E402
from cellplex.specialists.data import Windows, predict_cells  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['eva', 'gigatime'])
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--zeroshot', action='store_true')
    ap.add_argument('--epochs', type=float, default=None, help='passes over the training tissue (required for fine-tuning)')
    ap.add_argument('--config', default=f'{REPO}/configs/specialists.yaml')
    ap.add_argument('--ckpt', default='', help='starting weights (default: the released weights from the Hugging Face hub)')
    ap.add_argument('--bs', type=int, default=None, help='micro-batch (default: configs/specialists.yaml)')
    ap.add_argument('--accum', type=int, default=None, help='gradient accumulation (default: configs/specialists.yaml)')
    ap.add_argument('--n_val', type=int, default=None, help='validations, evenly spaced over the maximum steps')
    ap.add_argument('--patience', type=int, default=0, help='stop after this many validations without a gain > --min_delta (0: off)')
    ap.add_argument('--min_delta', type=float, default=0.002)
    ap.add_argument('--max_hours', type=float, default=0, help='training time budget; validate and stop when reached (0: off)')
    ap.add_argument('--eval_only', action='store_true', help='skip training: test the run\'s best_model.pt')
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--max_steps', type=int, default=0, help='smoke test (also: 2 test units, 8 blocks per slide)')
    ap.add_argument('--wandb', default='disabled', choices=['online', 'offline', 'disabled'])
    ap.add_argument('--wandb_project', default='cellplex')
    a = ap.parse_args()
    if a.epochs is None and not (a.zeroshot or a.eval_only or a.max_steps):
        ap.error('--epochs is required for fine-tuning')
    C = yaml.safe_load(open(a.config))
    R = C[a.model]
    a.bs, a.accum, a.n_val = a.bs or R['micro_batch'], a.accum or R['accum'], a.n_val or C['n_val']
    eff = a.bs * a.accum
    tag = f'{a.model}_{"zeroshot" if a.zeroshot else "ft"}'
    out = run_dir(a.cohort, tag, a.fold) + ('_smoke' if a.max_steps else '')
    os.makedirs(f'{out}/test', exist_ok=True)
    log_f = open(f'{out}/train.log', 'a')

    def log(msg):
        msg = f'{time.strftime("%H:%M:%S")} {msg}'; print(msg, flush=True); log_f.write(msg + '\n'); log_f.flush()

    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device('cuda')
    meta = load_meta(a.cohort)
    proteins, m0 = meta['proteins'], meta['mpp']
    U = fold_units(a.cohort, a.fold)
    tiles = load_tiles(a.cohort)
    M = Specialist(a.model, a.cohort, proteins, m0, a.zeroshot, dev, a.ckpt or None)
    scored = [proteins[j] for j in M.keep]
    fov = M.W * M.mpp
    area = (a.epochs or 0) * int(tiles.unit.isin(U['train']).sum()) * (TILE * m0) ** 2   # um^2 of training tissue
    n_win = int(area / fov ** 2)
    steps = n_win // eff if not a.max_steps else a.max_steps
    val_blocks = C['val_blocks'] if meta['reader'] == 'wsi' else None
    cfg = dict(model=a.model, zeroshot=a.zeroshot, cohort=a.cohort, fold=a.fold, proteins=scored, fov_um=fov,
               window_px=M.W, native_crop_px=M.L, target_mpp=M.mpp, train_windows=n_win, steps=steps, batch=eff,
               micro_batch=a.bs, accum=a.accum, units={k: sorted(v) for k, v in U.items()})
    json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
    wb = wandb_run('disabled' if a.max_steps else a.wandb, project=a.wandb_project, dir=out,
                   id=f'{a.cohort}_{tag}_fold{a.fold}', group=f'{a.cohort}_{tag}',
                   config={k: v for k, v in cfg.items() if k != 'units'})
    log(f'{a.cohort} fold {a.fold} {tag}: {len(scored)} proteins, units {[len(U[k]) for k in ("train", "val", "test")]}, '
        f'FOV {fov:.0f} um ({M.W} px at {M.mpp} um/px = {M.L} native px), {n_win:,} windows = {steps} steps x {eff} '
        f'({a.bs} x {a.accum} accumulation)')
    best, step, stop = -1.0, 0, 'zeroshot' if a.zeroshot else ''
    if a.eval_only:
        ck = torch.load(f'{out}/best_model.pt', map_location='cpu', weights_only=False)
        M.load_state_dict(ck['state_dict']); best, step, stop = ck['val_pcc'], ck['step'], 'eval only'
        del ck
        log(f'eval only: best_model.pt from step {step} (val {best:.4f})')
    elif not a.zeroshot:
        params = M.parameters()
        lr = R['lr'] * math.sqrt(eff / R['lr_ref_batch'])
        warm = R['warmup_steps']
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=R['weight_decay'])
        ds = Windows(a.cohort, sorted(U['train']), steps * eff, M.L, M.W)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4, worker_init_fn=lambda _: cv2.setNumThreads(1))
        val_at = set(int(round(steps * (k + 1) / a.n_val)) for k in range(a.n_val))
        M.net.train()
        t0, run, micro, bad = time.time(), [], 0, 0
        opt.zero_grad(set_to_none=True)
        for img, t, m in dl:
            loss = M.train_loss(img.to(dev, non_blocking=True), t.to(dev, non_blocking=True), m.to(dev, non_blocking=True))
            (loss / a.accum).backward()
            run.append(loss.item()); micro += 1
            if micro % a.accum:
                continue
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g in opt.param_groups:
                g['lr'] = lr * f
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log(f'step {step}/{steps} loss {np.mean(run):.5f} lr {lr * f:.2e} {step * eff / (time.time() - t0):.1f} win/s '
                    f'mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB')
                wb.log({'train/loss': float(np.mean(run)), 'train/lr': lr * f}, step=step); run = []
            over_time = bool(a.max_hours) and time.time() - t0 > a.max_hours * 3600
            if step in val_at or over_time:
                res = predict_cells(M, a.cohort, sorted(U['val']), dev, max_blocks=8 if a.max_steps else val_blocks, log=log)
                v = float(np.mean([slide_gene_pcc(p, t) for _, p, t in res.values()]))
                log(f'== val step {step}: protein PCC {v:.4f} (best {best:.4f})'); wb.log({'val/pcc': v}, step=step)
                bad = 0 if v > best + a.min_delta else bad + 1               # early stopping counts only real gains
                if v > best:
                    best = v
                    torch.save({'state_dict': M.state_dict(), 'step': step, 'val_pcc': v, 'proteins': scored,
                                'config': cfg}, f'{out}/best_model.pt.tmp')
                    os.replace(f'{out}/best_model.pt.tmp', f'{out}/best_model.pt')      # atomic: a cancel never truncates it
                if a.patience and bad >= a.patience:
                    stop = f'early stop: no gain > {a.min_delta} in {a.patience} validations'
                elif over_time:
                    stop = f'time budget {a.max_hours} h'
                if stop:
                    log(f'{stop} (step {step}/{steps}, best val {best:.4f})')
                    break
            if step >= steps:
                stop = 'max steps'
                break
        M.load_state_dict(torch.load(f'{out}/best_model.pt', map_location=dev, weights_only=False)['state_dict'])
    test = sorted(U['test'])[:2] if a.max_steps else sorted(U['test'])           # smoke: 2 units, 8 blocks
    res = predict_cells(M, a.cohort, test, dev, max_blocks=8 if a.max_steps else None, log=log)
    S, summ = save_test(out, res, scored, {'val_pcc': best, 'stopped_at_step': step, 'stop_reason': stop,
                                           'n_proteins': len(scored)})
    for k, v in summ.items():
        wb.summary[k] = v
    wb.finish()
    log(f'TEST {tag} fold {a.fold}: {summ["n_test_samples"]} samples, {len(scored)} proteins, PCC {summ["test_pcc"]:.4f} '
        f'+- {summ["test_pcc_sem"]:.4f}, Spearman {summ["test_spearman"]:.4f}')


if __name__ == '__main__':
    main()
