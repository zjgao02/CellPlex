"""Panel markers of the CellPlex cohorts -> genes, used to request markers from the specialist models (Eva by marker
name or gene embedding, GigaTIME by output channel). NORM: a marker name lower-cased with non-alphanumerics removed."""
import re

NORM = lambda x: re.sub(r'[^a-z0-9]', '', x.lower())   # noqa: E731
GENE = {'cd79a': 'CD79A', 'foxp3': 'FOXP3', 'gata3': 'GATA3', 'cd8': 'CD8A', 'p53': 'TP53', 'pax5': 'PAX5', 'cd21': 'CR2',
        'pdl1': 'CD274', 'ki67': 'MKI67', 'cd45': 'PTPRC', 'cd30': 'TNFRSF8', 'cd2': 'CD2', 'hladr': 'HLA-DRA', 'cd5': 'CD5',
        'pd1': 'PDCD1', 'cd45ra': 'PTPRC', 'cd4': 'CD4', 'cd56': 'NCAM1', 'muc1ema': 'MUC1', 'muc1': 'MUC1', 'bcl2': 'BCL2',
        'lag3': 'LAG3', 'cd71': 'TFRC', 'cd25': 'IL2RA', 'vista': 'VSIR', 'cd11c': 'ITGAX', 'egfr': 'EGFR', 'cd1a': 'CD1A',
        'nakatpase': 'ATP1A1', 'cd44': 'CD44', 'ido1': 'IDO1', 'cd38': 'CD38', 'cd16': 'FCGR3A', 'irf4': 'IRF4',
        'cd20': 'MS4A1', 'cd7': 'CD7', 'cd3': 'CD3E', 'cd3e': 'CD3E', 'granzymeb': 'GZMB', 'ccr4cd194': 'CCR4', 'cd194': 'CCR4',
        'cd57': 'B3GAT1', 'vimentin': 'VIM', 'cd68': 'CD68', 'icos': 'ICOS', 'cytokeratin': 'KRT8', 'cd34': 'CD34',
        'cd163': 'CD163', 'sma': 'ACTA2', 'asma': 'ACTA2', 'cd45ro': 'PTPRC', 'cd11b': 'ITGAM', 'collageniv': 'COL4A1',
        'cd31': 'PECAM1', 'cd134ox40': 'TNFRSF4', 'cd15': 'FUT4', 'cd138': 'SDC1', 'epcam': 'EPCAM', 'mmp9': 'MMP9',
        'mmp12': 'MMP12', 'podoplanin': 'PDPN', 'cdx2': 'CDX2', 'tbet': 'TBX21', 'betacatenin': 'CTNNB1',
        'synaptophysin': 'SYP', 'gfap': 'GFAP', 'chromogranina': 'CHGA', 'ngfr': 'NGFR', 'cd90': 'THY1', 'spink2': 'SPINK2',
        'cd117': 'KIT', 'foxc1': 'FOXC1', 'cd61': 'ITGB3', 'cd49f': 'ITGA6', 'vecadherin': 'CDH5', 'adiponectin': 'ADIPOQ',
        'cd123': 'IL3RA', 'cd14': 'CD14', 'plp1': 'PLP1', 'cd10': 'MME', 'tryptase': 'TPSAB1', 'tgfb1': 'TGFB1',
        'vcam1': 'VCAM1', 'cd146': 'MCAM', 'cd33': 'CD33', 'gata1': 'GATA1', 'cxcl12': 'CXCL12', 'glycophorinc': 'GYPC',
        'hif1a': 'HIF1A', 'mpo': 'MPO', 'oxphos': 'NDUFB8', 'cd141': 'THBD', 'cxcr4': 'CXCR4',
        'panck': 'KRT8', 'arginase1': 'ARG1', 'ca9': 'CA9', 'cd206': 'MRC1', 'fap': 'FAP', 'folr2': 'FOLR2',
        'isg15': 'ISG15', 'lyve1': 'LYVE1', 'pdgfrb': 'PDGFRB', 'spp1': 'SPP1',
        'mct': 'TPSAB1',                                  # MCT = mast cell tryptase (BC/CRC, HNSCC-LN)
        'cd69': 'CD69', 'p63': 'TP63', 'tcrgd': 'TRDC', 'il10': 'IL10', 'tenascinc': 'TNC', 'pdpn': 'PDPN',
        'bcatenin': 'CTNNB1', 'periostin': 'POSTN', 'cd73': 'NT5E',
        # glioma
        'mhcii': 'HLA-DRA', 'fn1': 'FN1', 'sox10': 'SOX10', 'vim': 'VIM', 'cd279': 'PDCD1', 'dcx': 'DCX', 'dcxv2': 'DCX',
        'sox4': 'SOX4', 'neun': 'RBFOX3', 'aqp4': 'AQP4', 'olig2': 'OLIG2', 'glut1': 'SLC2A1', 'pdgfra': 'PDGFRA',
        'chi3l1': 'CHI3L1', 'bcan': 'BCAN', 'ndrg1': 'NDRG1', 'gap43': 'GAP43', 'apoe': 'APOE', 'apod': 'APOD',
        'sox2': 'SOX2', 'snap25': 'SNAP25', 'mt1mmp': 'MMP14', 's100b': 'S100B', 'map2': 'MAP2', 'cd19': 'CD19',
        'il1b': 'IL1B', 'tnc': 'TNC', 'cd83': 'CD83', 'cspg5': 'CSPG5', 'cd24': 'CD24', 'sparc': 'SPARC', 'ng2': 'CSPG4',
        'stmn1': 'STMN1', 'ncan': 'NCAN', 'tubb3': 'TUBB3', 'vcan': 'VCAN', 'postn': 'POSTN', 'p2y12': 'P2RY12',
        'col1': 'COL1A1', 'hpln1': 'HAPLN1', 'hspg2': 'HSPG2', 'tnr': 'TNR', 'sox9': 'SOX9', 'flt1': 'FLT1',
        'ptprz1': 'PTPRZ1', 'atrx': 'ATRX', 'acan': 'ACAN',
        'ecadherin': 'CDH1'}
