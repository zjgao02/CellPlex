"""Eva (Liu et al. 2025, github.com/YAndrewL/Eva; weights yandrewl/Eva, Eva_ft.ckpt): H&E -> multiplex virtual staining.

Setup of CellPlex: H&E only, 1 - RGB/255 as the channels HECHA1-3, at 0.5 um/px in 224 px windows; every panel protein
is requested by an Eva marker name. Markers missing from Eva's marker table are added with their gene, so that Eva
uses the gene's GenePT embedding; a protein whose gene has neither an Eva name nor a GenePT embedding is not predicted.
Eva's code is installed from its repository (docs/specialists.md): $EVA_REPO is the clone, and its working directory
($EVA_WORKDIR, default $EVA_REPO) holds marker_embeddings/GenePT_embedding.pkl.
"""
import os, pickle, sys

import numpy as np
import torch
import torch.nn.functional as F

from .markers import GENE, NORM

HE = ['HECHA1', 'HECHA2', 'HECHA3']
WIN = 224
MPP = 0.5
# fixed marker names of two panels (cohort -> {panel marker: Eva marker})
NAME_MULTITUMOR = {'CD79a': 'CD79a', 'FOXP3': 'FoxP3', 'CDX2': 'CDX2', 'CD8': 'CD8', 'p53': 'p53', 'GATA3': 'GATA3', 'CD21': 'CD21',
                   'PD-L1': 'PDL1', 'Ki67': 'Ki67', 'CD45': 'CD45', 'CD30': 'CD30', 'PAX5': 'PAX5', 'HLA-DR': 'HLA-DR', 'CD5': 'CD5',
                   'CD2': 'CD2', 'CD45RA': 'CD45RA', 'CD4': 'CD4', 'PD-1': 'PD1', 'MUC-1': 'MUC1', 'BCL2': 'BCL2', 'CD56': 'CD56',
                   'Cytokeratin 7': 'CK7', 'CD25': 'CD25', 'VISTA': 'VISTA', 'Hep-Par-1': 'HepPar1', 'CD11c': 'CD11c', 'IRF4': 'IRF4',
                   'CD20': 'CD20', 'EGFR': 'EGFR', 'IDO-1': 'IDO1', 'Granzyme B': 'GranzymeB', 'Melan A': 'MelanA', 'OX-40': 'OX40',
                   'Vimentin': 'Vimentin', 'Synaptophysin': 'Synaptophysin', 'CD117 (c-kit)': 'CD117', 'Na-K-ATPase': 'NaKATPase',
                   'CD194': 'CCR4', 'CD57': 'CD57', 'Pan-Cytokeratin': 'PanCK', 'CD68': 'CD68', 'CD3': 'CD3e', 'SMA': 'aSMA',
                   'CD34': 'CD34', 'EpCAM': 'EpCAM', 'Collagen IV': 'CollagenIV', 'CD45RO': 'CD45RO', 'Podoplanin': 'Podoplanin',
                   'CD15': 'CD15', 'CD7': 'CD7', 'CD163': 'CD163', 'Chromogranin A': 'ChromograninA', 'CD31': 'CD31', 'CD123': 'CD123',
                   'MMP9': 'MMP9', 'CD138': 'CD138', 'CD38': 'CD38'}
NAME_ORION = {'CD31': 'CD31', 'CD45': 'CD45', 'CD68': 'CD68', 'CD4': 'CD4', 'FOXP3': 'FoxP3', 'CD8': 'CD8',
              'CD45RO': 'CD45RO', 'CD20': 'CD20', 'PD-L1': 'PDL1', 'CD3e': 'CD3e', 'CD163': 'CD163',
              'E-cadherin': 'ECad', 'PD-1': 'PD1', 'Ki67': 'Ki67', 'Pan-CK': 'PanCK', 'SMA': 'aSMA'}
FIXED = {'multitumor': NAME_MULTITUMOR, 'orion_crc': NAME_ORION}
# Eva names missing from its marker table -> gene (multitumor panel)
ADD = {'CD79a': 'CD79A', 'PAX5': 'PAX5', 'MUC1': 'MUC1', 'CK7': 'KRT7', 'HepPar1': 'CPS1', 'MelanA': 'MLANA',
       'NaKATPase': 'ATP1A1', 'CCR4': 'CCR4'}
# normalised panel names with a fixed Eva name
SPECIAL = {'cytokeratin': 'PanCK', 'sma': 'aSMA', 'asma': 'aSMA', 'cd45': 'CD45', 'cd45ro': 'CD45RO', 'cd45ra': 'CD45RA',
           'panck': 'PanCK', 'mct': 'Tryptase'}


def setup():
    """put Eva's code on the path and enter its working directory (Eva reads its marker embeddings relative to it)"""
    repo = os.environ.get('EVA_REPO')
    assert repo, 'set EVA_REPO to a clone of github.com/YAndrewL/Eva (docs/specialists.md)'
    sys.path.insert(0, repo)
    os.chdir(os.environ.get('EVA_WORKDIR', repo))
    return repo


def checkpoint():
    from huggingface_hub import hf_hub_download
    return os.environ.get('EVA_CKPT') or hf_hub_download('yandrewl/Eva', 'Eva_ft.ckpt')


def names(cohort, proteins):
    """(kept protein indices, Eva marker names); adds the new names to Eva's marker table (call before building
    the model, which reads the table)"""
    import utils.constant as UC
    M = UC.marker_to_gene
    if cohort in FIXED:
        for k, g in ADD.items():
            M.setdefault(k, g)
        return list(range(len(proteins))), [FIXED[cohort][p] for p in proteins]
    genept = pickle.load(open('marker_embeddings/GenePT_embedding.pkl', 'rb'))
    by_norm = {NORM(k): k for k in M}
    keep, out = [], []
    for j, p in enumerate(proteins):
        n = NORM(p)
        g = GENE[n]
        c = [k for k, v in M.items() if v == g]
        if n in SPECIAL:
            name = SPECIAL[n]
        elif n in by_norm:
            name = by_norm[n]
        elif len(c) == 1 or (c and g not in genept):
            name = c[0]
        elif g in genept:
            name = f'cellplex_{n}'; M[name] = g
        else:
            continue
        keep.append(j); out.append(name)
    return keep, out


def build(device='cuda', ckpt=None):
    from omegaconf import OmegaConf
    from Eva.eva import EvaMAE
    return EvaMAE.from_checkpoint(ckpt or checkpoint(), OmegaConf.load(f'{os.environ["EVA_REPO"]}/config.yaml'), device=device)


@torch.no_grad()
def sliding(model, x, out_markers, stride=112, bs=8):
    """x (1, 3, h, w) = 1 - RGB/255 at 0.5 um/px -> (len(out_markers), h, w), overlapping windows averaged."""
    _, _, h, w = x.shape
    ph = WIN if h <= WIN else int(np.ceil((h - WIN) / stride)) * stride + WIN
    pw = WIN if w <= WIN else int(np.ceil((w - WIN) / stride)) * stride + WIN
    xp = F.pad(x, (0, pw - w, 0, ph - h), mode='replicate')
    acc = torch.zeros(len(out_markers), ph, pw, device=x.device)
    cnt = torch.zeros(1, ph, pw, device=x.device)
    pos = [(i, j) for i in range(0, ph - WIN + 1, stride) for j in range(0, pw - WIN + 1, stride)]
    mask = torch.zeros(len(HE), (WIN // model.token_size) ** 2, device=x.device)
    for k in range(0, len(pos), bs):
        chunk = pos[k:k + bs]
        batch = torch.cat([xp[:, :, i:i + WIN, j:j + WIN] for i, j in chunk], 0).permute(0, 2, 3, 1)
        rec, _, _ = model(batch, marker_in=[HE] * len(chunk), marker_out=[out_markers] * len(chunk), infer_mask=mask)
        for (i, j), pp in zip(chunk, rec.permute(0, 3, 1, 2).float()):
            acc[:, i:i + WIN, j:j + WIN] += pp
            cnt[:, i:i + WIN, j:j + WIN] += 1
    return (acc / cnt)[:, :h, :w]
