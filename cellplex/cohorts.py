"""The seven cohorts of CellPlex.

mpp: resolution of the H&E as used by the models (native, no resizing); reader: 'wsi' (pyramidal TIFF read with
OpenSlide) or 'npy' (one uint8 (H, W, 3) array per core or region); unit: the split unit; folds: the folds that are
evaluated (five-fold cross-validation, or one fixed split for the two cohorts with the largest tissue area).
"""
TILE = 224          # non-overlapping 224 x 224 tiles at native resolution
N_FOLDS = 5

COHORTS = {
    'orion_crc':   dict(name='Orion-CRC',   mpp=0.325,    reader='wsi', unit='patient', folds=[0]),
    'crc_tma':     dict(name='CRC-TMA',     mpp=0.377442, reader='npy', unit='core',    folds=list(range(5))),
    'multitumor':  dict(name='Multi-tumor', mpp=0.377442, reader='npy', unit='core',    folds=list(range(5))),
    'bc_crc':      dict(name='BC/CRC',      mpp=0.5072,   reader='npy', unit='region',  folds=list(range(5))),
    'hnscc_ln':    dict(name='HNSCC-LN',    mpp=0.377442, reader='npy', unit='patient', folds=list(range(5))),
    'bone_marrow': dict(name='Bone marrow', mpp=0.2485,   reader='wsi', unit='patient', folds=list(range(5))),
    'glioma':      dict(name='Glioma',      mpp=0.2485,   reader='wsi', unit='patient', folds=[0]),
}
