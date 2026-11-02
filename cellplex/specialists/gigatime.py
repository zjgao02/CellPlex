"""GigaTIME (Valanarasu et al. 2026, github.com/prov-gigatime/GigaTIME; weights prov-gigatime/GigaTIME, model.pth):
H&E -> 21-protein virtual multiplex immunofluorescence (a NestedUNet).

Setup of CellPlex: ImageNet-normalised RGB at 0.25 um/px; inference in 256 px windows (stride 128, overlaps averaged,
sigmoid outputs); fine-tuning on 512 px crops processed as 4 windows of 256, with one output per panel protein whose
head is initialised from the same protein's GigaTIME channel where it has one (else the mean of its protein channels),
and GigaTIME's loss (0.5 BCE + 1 - Dice) on tissue pixels against the target coarsened 1/8 as in its training code.
Zero-shot GigaTIME is scored only on the panel proteins matching one of its channels. Its code is installed from its
repository (docs/specialists.md): $GIGATIME_REPO is the clone (scripts/archs.py).
"""
import os, sys

import numpy as np
import torch
import torch.nn.functional as F

from .markers import GENE, NORM

CHANNELS = ['DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34', 'CD68', 'CD16', 'CD11c', 'CD138',
            'CD20', 'CD3', 'CD8', 'PD-L1', 'CK', 'Ki67', 'Tryptase', 'Actin-D', 'Caspase3-D', 'PHH3-B', 'Transgelin']
NON_PROTEIN = ('DAPI', 'TRITC', 'Cy5')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WIN = 256
CROP = 512                                           # training crop at 0.25 um/px, split into 4 windows of WIN
MPP = 0.25
# multitumor panel marker -> GigaTIME channel(s) of the same protein
GT_SAME = {'PD-1': ['PD-1'], 'CD4': ['CD4'], 'CD34': ['CD34'], 'CD68': ['CD68'], 'CD11c': ['CD11c'], 'CD138': ['CD138'],
           'CD20': ['CD20'], 'CD3': ['CD3'], 'CD8': ['CD8'], 'PD-L1': ['PD-L1'], 'Pan-Cytokeratin': ['CK'], 'Ki67': ['Ki67']}
# GigaTIME channel -> gene (other panels are matched through the gene of the marker)
GT_GENE = {'PD-1': 'PDCD1', 'CD14': 'CD14', 'CD4': 'CD4', 'T-bet': 'TBX21', 'CD34': 'CD34', 'CD68': 'CD68', 'CD16': 'FCGR3A',
           'CD11c': 'ITGAX', 'CD138': 'SDC1', 'CD20': 'MS4A1', 'CD3': 'CD3E', 'CD8': 'CD8A', 'PD-L1': 'CD274', 'CK': 'KRT8',
           'Ki67': 'MKI67', 'Tryptase': 'TPSAB1'}


def setup():
    repo = os.environ.get('GIGATIME_REPO')
    assert repo, 'set GIGATIME_REPO to a clone of github.com/prov-gigatime/GigaTIME (docs/specialists.md)'
    sys.path.insert(0, f'{repo}/scripts')
    return repo


def checkpoint():
    from huggingface_hub import hf_hub_download
    return os.environ.get('GIGATIME_CKPT') or hf_hub_download('prov-gigatime/GigaTIME', 'model.pth')


def columns(cohort, proteins):
    """(kept protein indices, GigaTIME channel indices) of the panel proteins GigaTIME measures"""
    if cohort == 'multitumor':
        keep = [j for j, p in enumerate(proteins) if p in GT_SAME]
        return keep, [CHANNELS.index(GT_SAME[proteins[j]][0]) for j in keep]
    g2c = {g: c for c, g in GT_GENE.items()}
    keep = [j for j, p in enumerate(proteins) if GENE.get(NORM(p)) in g2c]
    return keep, [CHANNELS.index(g2c[GENE[NORM(proteins[j])]]) for j in keep]


def build(cohort, proteins, zeroshot, ckpt=None):
    """GigaTIME with its released weights; for fine-tuning a new output layer, one channel per panel protein.
    -> (model, kept protein indices, output columns to keep or None)"""
    import archs
    m = archs.gigatime(len(CHANNELS), 3)
    m.load_state_dict(torch.load(ckpt or checkpoint(), map_location='cpu', weights_only=True))
    if zeroshot:                                                      # only proteins GigaTIME measures
        keep, cols = columns(cohort, proteins)
        return m, keep, cols
    old = m.final
    new = torch.nn.Conv2d(old.in_channels, len(proteins), 1)
    prot_ch = [i for i, c in enumerate(CHANNELS) if c not in NON_PROTEIN]
    with torch.no_grad():
        same = dict(zip(*columns(cohort, proteins)))                      # protein index -> GigaTIME channel
        for j in range(len(proteins)):
            ix = [same[j]] if j in same else prot_ch
            new.weight[j] = old.weight[ix].mean(0); new.bias[j] = old.bias[ix].mean(0)
    m.final = new
    return m, list(range(len(proteins))), None


@torch.no_grad()
def sliding(model, x, stride=128, bs=128):
    """x (1, 3, h, w) normalised -> (K, h, w) sigmoid probabilities (K = model outputs), overlapping windows averaged."""
    _, _, h, w = x.shape
    ph = WIN if h <= WIN else int(np.ceil((h - WIN) / stride)) * stride + WIN
    pw = WIN if w <= WIN else int(np.ceil((w - WIN) / stride)) * stride + WIN
    xp = F.pad(x, (0, pw - w, 0, ph - h), mode='replicate')
    acc = torch.zeros(model.final.out_channels, ph, pw, device=x.device)
    cnt = torch.zeros(1, ph, pw, device=x.device)
    pos = [(i, j) for i in range(0, ph - WIN + 1, stride) for j in range(0, pw - WIN + 1, stride)]
    for k in range(0, len(pos), bs):
        chunk = pos[k:k + bs]
        batch = torch.cat([xp[:, :, i:i + WIN, j:j + WIN] for i, j in chunk], 0)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            pr = torch.sigmoid(model(batch).float())
        for (i, j), pp in zip(chunk, pr):
            acc[:, i:i + WIN, j:j + WIN] += pp
            cnt[:, i:i + WIN, j:j + WIN] += 1
    return (acc / cnt)[:, :h, :w]


def windows(x):
    """(B, C, 512, 512) -> (4B, C, 256, 256): the crop's 4 windows, as GigaTIME runs them."""
    B, C = x.shape[:2]
    x = x.reshape(B, C, 2, WIN, 2, WIN).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B * 4, C, WIN, WIN)


def prep(img, t, m, dev, mean, std):
    """uint8 batch -> normalised windows, coarsened [0, 1] target, tissue mask (all (4B, ., 256, 256))."""
    x = (img.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255. - mean) / std
    y = F.interpolate(t.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255., size=CROP, mode='bilinear',
                      align_corners=False)
    y = F.interpolate(F.interpolate(y, scale_factor=1 / 8, mode='bilinear', align_corners=False), size=CROP,
                      mode='bilinear', align_corners=False)                  # coarse target of GigaTIME's training
    k = F.interpolate(m.to(dev, non_blocking=True)[:, None].float(), size=CROP, mode='bilinear', align_corners=False) > 0.99
    return windows(x), windows(y), windows(k.float())[:, 0] > 0.5


def bce_dice(logits, y, k):
    """GigaTIME's BCEDiceLoss (0.5 BCE + 1 - Dice, Dice per sample over all channels), tissue pixels only."""
    w = k[:, None].float().expand_as(y)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logits) * w
    yt = y * w
    dice = (2 * (p * yt).flatten(1).sum(1) + 1e-5) / (p.flatten(1).sum(1) + yt.flatten(1).sum(1) + 1e-5)
    keep = k.flatten(1).any(1)                                               # windows with tissue
    return 0.5 * bce + (1 - dice[keep].mean()), bce
