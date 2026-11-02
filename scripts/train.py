"""Train and test CELLO with one pathology foundation model on one fold of one cohort, on one GPU.

Recipe (configs/cello.yaml, CELLO's defaults): CELLO cross-attention model, Huber cell loss + 0.5 x patch loss, AdamW
(encoder 5e-6 -> 2e-5, wd 0.01; head 2e-4, wd 0.05), cosine schedule, fp16 autocast + GradScaler, grad clip 1.0,
encoder frozen for the first 10% of training, linear encoder-LR warm-up over the next 30%, then fully fine-tuned;
batch 64 tiles; <= 256 cells per training tile. The number of epochs is given with --epochs. With the per-step
schedule (--step_schedule, default on for runs of <= 20 epochs) the same ratios are applied per step and validation
runs every epoch, otherwise at epoch 0 and every 5 epochs. Validation score = mean
over validation samples of the sample's mean protein PCC (every cell of every foreground tile); the best checkpoint is
kept. The best checkpoint then predicts every cell of the test units -> {out}/test (cellplex/engine.py).

  python scripts/train.py --cohort multitumor --fold 0 --encoder hoptimus0 --epochs N
  python scripts/train.py --cohort multitumor --fold 0 --encoder hoptimus0 --eval_only      test best_model.pt again
  python scripts/train.py --cohort multitumor --fold 0 --encoder uni2 --max_steps_per_epoch 3 --epochs 1   smoke test
Encoders: uni2 (MahmoodLab/UNI2-h), virchow2 (paige-ai/Virchow2), hoptimus0 (bioptimus/H-optimus-0); their weights
are downloaded from the Hugging Face hub (request access on the model pages and log in with `huggingface-cli login`).
Output: $CELLPLEX_RUNS/{cohort}/{encoder}/fold{k}/ (config.json, train.log, best_model.pt, latest_model.pt, test/).
"""
import argparse, json, logging, math, os, random, sys, time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellplex.cello_utils import collate_fn, slide_gene_pcc  # noqa: E402
from cellplex.data import TileDS, split_datasets  # noqa: E402
from cellplex.engine import predict, save_test, wandb_run  # noqa: E402
from cellplex.model import CELLO  # noqa: E402
from cellplex.paths import REPO, run_dir  # noqa: E402


def load_config(path):
    C = yaml.safe_load(open(path))
    return {**C['model'], **C['loss'], **C['train']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--encoder', required=True, help='uni2 | virchow2 | hoptimus0')
    ap.add_argument('--config', default=f'{REPO}/configs/cello.yaml')
    ap.add_argument('--out', default=None, help='default: $CELLPLEX_RUNS/{cohort}/{encoder}/fold{k}')
    ap.add_argument('--epochs', type=int, default=None, help='training epochs (required unless --eval_only)')
    ap.add_argument('--batch_size_per_gpu', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--max_steps_per_epoch', type=int, default=0, help='smoke test only (also: 100 val / test tiles per sample)')
    ap.add_argument('--step_schedule', type=int, default=None, choices=[0, 1],
                    help='1: CELLO ratios per step (frozen first 10%% of steps, encoder-LR warm-up over the next 30%%, '
                         'head cosine per step) and validation every epoch; default: on when epochs <= 20')
    ap.add_argument('--no_resume', action='store_true')
    ap.add_argument('--eval_only', action='store_true', help='skip training: test the run\'s best_model.pt')
    ap.add_argument('--wandb', default='disabled', choices=['online', 'offline', 'disabled'])
    ap.add_argument('--wandb_project', default='cellplex')
    a = ap.parse_args()
    if a.epochs is None and not a.eval_only:
        ap.error('--epochs is required for training')
    cfg = dict(load_config(a.config), vision_encoder=a.encoder)
    for k in ('epochs', 'batch_size_per_gpu', 'num_workers'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    out = a.out or run_dir(a.cohort, a.encoder, a.fold)
    os.makedirs(f'{out}/test', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(f'{out}/train.log'), logging.StreamHandler()])
    log = logging.getLogger('cellplex').info
    dev = torch.device('cuda')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])

    D, U = split_datasets(a.cohort, a.fold, cfg['max_cells_per_patch'])
    if a.max_steps_per_epoch:                                        # smoke test: 100 val / test tiles per sample
        for k in ('val', 'test'):
            D[k] = TileDS(a.cohort, D[k].tiles.groupby('sample', group_keys=False).head(100), max_cells_per_patch=None)
    proteins = D['train'].proteins
    cfg.update(cohort=a.cohort, fold=a.fold, units={k: sorted(v) for k, v in U.items()}, proteins=proteins)
    log(f'{a.cohort} fold {a.fold} {a.encoder}: tiles train {len(D["train"])} val {len(D["val"])} test {len(D["test"])}; '
        f'units {[len(U[k]) for k in ("train", "val", "test")]}; {len(proteins)} proteins; '
        f'GPU {torch.cuda.get_device_name()}')

    wb = wandb_run(a.wandb if not a.max_steps_per_epoch else 'disabled', project=a.wandb_project, dir=out,
                   id=f'{a.cohort}_{a.encoder}_fold{a.fold}', group=f'{a.cohort}_{a.encoder}',
                   config={k: v for k, v in cfg.items() if k != 'units'})
    model = CELLO(proteins, cfg, pretrained_encoder=not a.eval_only).to(dev)
    model.union_genes = proteins
    pmap = {g: i for i, g in enumerate(proteins)}
    log(f'parameters {sum(p.numel() for p in model.parameters()):,}; backbone dim {model.backbone_dim}, '
        f'token grid {model.token_grid}, prefix tokens {model.num_prefix_tokens}')
    t0 = time.time()

    if not a.eval_only:
        bs = cfg['batch_size_per_gpu']
        loader = DataLoader(D['train'], batch_size=bs, shuffle=True, num_workers=cfg['num_workers'], pin_memory=True,
                            collate_fn=collate_fn, persistent_workers=True, prefetch_factor=4, drop_last=False)
        enc_ids = {id(p) for p in model.encoder.parameters()}
        enc_params = [p for p in model.parameters() if id(p) in enc_ids]
        head_params = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{'params': enc_params, 'lr': cfg['lr_encoder'], 'weight_decay': cfg['weight_decay_encoder']},
                                 {'params': head_params, 'lr': cfg['lr_head'], 'weight_decay': cfg['weight_decay_head']}])
        scaler = torch.amp.GradScaler('cuda')
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-8)
        start, best = 0, -1.0
        latest = f'{out}/latest_model.pt'
        if not a.no_resume and os.path.exists(latest):
            ck = torch.load(latest, map_location='cpu', weights_only=False)   # on the CPU: it would stay resident on the GPU
            model.load_state_dict(ck['model_state_dict']); opt.load_state_dict(ck['optimizer_state_dict'])
            sched.load_state_dict(ck['scheduler_state_dict']); start, best = ck['epoch'] + 1, ck['best_pcc']
            del ck
            log(f'resumed at epoch {start} (best {best:.4f})')

        E = cfg['epochs']
        freeze_e, warm_e = int(E * cfg['freeze_encoder_ratio']), int(E * cfg['gradual_unfreeze_ratio'])
        step_mode = bool(a.step_schedule if a.step_schedule is not None else E <= 20)
        spe = min(len(loader), a.max_steps_per_epoch or len(loader))
        total = E * spe
        freeze_s, warm_s = int(total * cfg['freeze_encoder_ratio']), int(total * cfg['gradual_unfreeze_ratio'])
        cfg['step_schedule'] = step_mode
        if step_mode:
            cfg['val_every'] = 1
            log(f'epochs {E} x {spe} steps, per-step schedule: encoder frozen for {freeze_s} steps, warm-up {warm_s} '
                f'steps; head cosine per step; validation every epoch; batch {bs}')
        else:
            log(f'epochs {E}: encoder frozen <= {freeze_e}, warm-up {warm_e}; batch {bs}; {len(loader)} steps / epoch')
        json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)

        def save(path, epoch):
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': sched.state_dict(), 'best_pcc': best, 'config': cfg,
                        'union_genes': proteins}, path + '.tmp')
            os.replace(path + '.tmp', path)                          # atomic: a killed job never leaves a truncated file

        def set_trainable(on):
            for p in model.encoder.parameters():
                p.requires_grad = on

        for epoch in range(start, E):
            model.train()
            flag = epoch > freeze_e
            if not step_mode:                                        # CELLO: per-epoch encoder schedule
                set_trainable(flag)
                if freeze_e < epoch <= freeze_e + warm_e:
                    opt.param_groups[0]['lr'] = (cfg['lr_encoder'] + (cfg['lr_encoder_max'] - cfg['lr_encoder'])
                                                 * (epoch - freeze_e) / warm_e)
                elif epoch > freeze_e + warm_e:
                    opt.param_groups[0]['lr'] = cfg['lr_encoder_max']
            run = []
            for step, batch in enumerate(loader):
                if a.max_steps_per_epoch and step >= a.max_steps_per_epoch:
                    break
                if step_mode:                                        # same ratios, per step
                    g = epoch * spe + step
                    flag = g >= freeze_s
                    if step == 0 or g == freeze_s:
                        set_trainable(flag)
                    if flag:
                        opt.param_groups[0]['lr'] = (cfg['lr_encoder'] + (cfg['lr_encoder_max'] - cfg['lr_encoder'])
                                                     * min(1.0, (g - freeze_s) / max(1, warm_s)))
                    opt.param_groups[1]['lr'] = 1e-8 + (cfg['lr_head'] - 1e-8) * 0.5 * (1 + math.cos(math.pi * g / total))
                with torch.amp.autocast('cuda'):
                    o = model.forward_batch(batch, pmap)
                    loss = cfg['l_gene'] * o['l_gene'] + cfg['l_pg'] * o['l_pg']
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(opt); scaler.update()
                run.append((loss.item(), o['l_gene'].item(), o['l_pg'].item()))
            m = np.mean(run, 0) if run else [np.nan] * 3
            log(f'epoch {epoch}: loss {m[0]:.4f} (cell {m[1]:.4f}, patch {m[2]:.4f}) enc lr {opt.param_groups[0]["lr"]:.2e} '
                f'head lr {opt.param_groups[1]["lr"]:.2e} enc {"on" if flag else "frozen"} '
                f'{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB')
            wb.log({'train/loss': m[0], 'train/cell_loss': m[1], 'train/patch_loss': m[2],
                    'train/lr_encoder': opt.param_groups[0]['lr'], 'train/lr_head': opt.param_groups[1]['lr']}, step=epoch)
            if epoch == 0 or (epoch + 1) % cfg['val_every'] == 0 or epoch == E - 1:
                res = predict(model, D['val'], dev, cfg['val_batch_size'], cfg['num_workers'], foreground=True)
                sc = {s: slide_gene_pcc(p, t) for s, (_, p, t) in res.items()}
                v = float(np.mean(list(sc.values()))) if sc else 0.0
                log(f'epoch {epoch}: val protein PCC {v:.4f} over {len(sc)} samples (best {best:.4f})')
                wb.log({'val/pcc': v}, step=epoch)
                if v > best:
                    best = v
                    save(f'{out}/best_model.pt', epoch)
                    log(f'  new best at epoch {epoch}')
            if not step_mode:
                sched.step()
            save(latest, epoch)

    ck = torch.load(f'{out}/best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    best_epoch, best_val = ck['epoch'], ck['best_pcc']
    del ck
    res = predict(model, D['test'], dev, cfg['val_batch_size'], cfg['num_workers'])
    S, summ = save_test(out, res, proteins, {'best_epoch': best_epoch, 'val_pcc': best_val})
    log(f'TEST (best epoch {best_epoch}, val {best_val:.4f}): {summ["n_test_samples"]} samples, PCC {summ["test_pcc"]:.4f} '
        f'+- {summ["test_pcc_sem"]:.4f}, Spearman {summ["test_spearman"]:.4f}; {time.time() - t0:.0f}s')
    for k, v in summ.items():
        wb.summary[k] = v
    wb.finish()


if __name__ == '__main__':
    main()
