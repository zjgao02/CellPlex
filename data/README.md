# CellPlex data

The processed cohorts are not distributed. This folder holds what is needed to rebuild them from the original
releases: the splits (`splits/{cohort}.csv`: unit, fold), the protein panels (`panels/{cohort}.csv`: marker name as
in the multiplex panel, canonical protein name), channel tables used by the processing scripts, and the download
lists (`urls/`). Please cite the original studies and follow their licences.

| cohort | study | release | licence | H&E public | samples used |
|---|---|---|---|---|---|
| `orion_crc` | Lin et al. 2023, Nat Cancer | AWS S3 `s3://lin-2023-orion-crc/data/` (public, no sign-in: `aws s3 ls --no-sign-request`) | not stated in the release | yes (registered to the Orion image) | 41 slides / 40 patients |
| `crc_tma` | Schürch et al. 2020, Cell | TCIA `CRC_FFPE-CODEX_CellNeighs`, doi:10.7937/TCIA.2020.FQN0-0326 | TCIA data usage policy (citation required) | yes | 130 of 140 cores |
| `multitumor` | Schürch et al. 2020, Cell | same TCIA collection | as above | yes | 66 cores |
| `bc_crc` | Matusiak et al. 2024, Cancer Discov | images from the authors (see below) | — | **no** | 45 of 46 regions |
| `hnscc_ln` | Haist et al. 2026, Cancer Cell | BioImage Archive S-BIAD1638 (doi:10.6019/S-BIAD1638); clinical table Mendeley Data doi:10.17632/t4k5kj3cxr.1 | CC0; CC BY 4.0 | yes | 388 of 390 cores / 79 patients |
| `bone_marrow` | Bandyopadhyay et al. 2024, Cell | Figshare+ collection doi:10.25452/figshare.plus.c.7174914 | CC0 | yes (registration to the H&E: see below) | 11 of 12 sections |
| `glioma` | Hoefflin et al. 2026, Cancer Cell | BioImage Archive S-BIAD2840 (doi:10.6019/S-BIAD2840); sample table Zenodo 21335411 | CC BY 4.0 | yes | 36 of 37 sections / 30 patients |

## Setup

```bash
export CELLPLEX_RAW=/path/to/raw          # one folder per cohort, laid out as below
export CELLPLEX_DATA=/path/to/processed   # written by scripts/preprocess/
export CELLVIT_DIR=/path/to/CellViT-Plus-Plus
```

Nuclei are segmented on the H&E with CellViT++ (https://github.com/TIO-IKIM/CellViT-Plus-Plus, commit 463c5c4,
model `CellViT-SAM-H-x40-AMP.pth`) at 0.25 µm/px, its native resolution: H&E scanned at a coarser resolution is
resampled to 0.25 µm/px by the cohort's `prep` step (one global pixel-centre convention,
`x_in = (x_out + 0.5) / sx - 0.5`, so the nuclei map back exactly), and slides scanned at 0.2485 µm/px are segmented
as they are. `scripts/preprocess/run_cellvit.py` runs CellViT++'s `detect_cells.py` with process counts sized to the
job's CPUs; `scripts/preprocess/run_cellvit.sbatch` is a SLURM example.

Every cohort then follows one protocol: cell region = nucleus expanded by 2 µm without entering neighbours; truth =
log1p of the mean multiplex intensity in the region, read after moving the regions into the multiplex frame (an
integer shift per core or region for the arrays, which are imaged on one pixel grid; a registration for the glioma
slides). A sample is excluded when its nuclear stain is brighter in the nucleus than in the 2 µm ring around it for
fewer than 80% of cells (H&E and multiplex still misaligned). Finally, for every cohort:

```bash
python scripts/preprocess/build_dataset.py --cohort {cohort}     # meta.json, cells/, tiles.csv, folds.csv
python scripts/preprocess/pixel_targets.py --cohort {cohort}     # only for fine-tuning Eva / GigaTIME
```

## Orion-CRC (`orion_crc`)

Download the registered H&E, the Orion image and `markers.csv` of every slide folder (`urls/orion_crc.txt`; the
Orion images are ~4.5 TB in total), keeping the folders:

```
$CELLPLEX_RAW/orion_crc/{CRC01..CRC40,CRC33_01,CRC33_02}/{*-registered.ome.tif, *-zlib.ome.tiff, markers.csv}
```

```bash
S=CRC03_Scan1                                                      # CRC33_01 / CRC33_02 -> CRC33_Scan1 / CRC33_Scan2
python scripts/preprocess/orion_crc.py grid --slide $S
python scripts/preprocess/orion_crc.py wsi --slide $S --workers 16
python scripts/preprocess/orion_crc.py cvin --slide $S --workers 16 --pieces --budget 3000
python scripts/preprocess/run_cellvit.py --model $CELLVIT_DIR/checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir $CELLPLEX_DATA/orion_crc/seg_raw/$S process_wsi --wsi_path $CELLPLEX_DATA/orion_crc/wsi_cv/$S.tif \
    --wsi_properties '{"slide_mpp": 0.25, "magnification": 40}'      # once per strip of wsi_cv_pieces/ when split
python scripts/preprocess/orion_crc.py seg --slide $S
python scripts/preprocess/orion_crc.py quant --slide $S --workers 16
```

Our processing read the same two images through 512 px tiles. The Orion channels of those tiles are identical to the
public image; the H&E pixels differ slightly (mean absolute difference 1.3 of 255, at JPEG block edges), and the set
of tissue tiles is defined here by an H&E saturation rule (on CRC03 it keeps 8,096 of our 8,103 tiles and adds 286
marginal ones), so nuclei and cell order will not be bit-identical.

## CRC-TMA and Multi-tumor (`crc_tma`, `multitumor`)

Download the TCIA collection with IBM Aspera (collection page, "Download"); only the best-focus planes are needed
(`urls/tcia_crc_ffpe_codex_cellneighs.txt` lists them with their MD5 sums, ~131 GB):

```
$CELLPLEX_RAW/crc_tma/{CRC_TMA_A,CRC_TMA_B}_HandE/bestFocus/reg###_X01_Y01_Z##.tif
$CELLPLEX_RAW/crc_tma/{CRC_TMA_A,CRC_TMA_B}_hyperstacks/{bestFocus/reg###_X01_Y01_Z##.tif, channelNames.txt, Experiment.json}
$CELLPLEX_RAW/multitumor/Multi-tumor_TMA_HandE/bestFocus/...,  Multi-tumor_TMA_hyperstacks/...
```

```bash
python scripts/preprocess/codex_tma.py prep --cohort crc_tma
sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/crc_tma/cellvit_filelist.csv $CELLPLEX_DATA/crc_tma/seg
python scripts/preprocess/codex_tma.py quant --cohort crc_tma     # the same three steps with --cohort multitumor
```

Samples: `A_reg001`..`B_reg070` and `reg001`..`reg070`. Excluded by QC (nucleus > ring): A_reg006 (0.745),
A_reg014 (0.748), A_reg024 (0.786), A_reg053 (0.727), A_reg058 (0.783), B_reg021 (0.500), B_reg022 (0.755),
B_reg042 (0.768), B_reg043 (0.761), B_reg058 (0.646). CRC-TMA is split by core (patient identifiers are in the study's
per-cell table, Mendeley Data doi:10.17632/mpjzbtfgfr.1).

## BC/CRC (`bc_crc`)

The images used (H&E montages and 8-bit CODEX montages on one pixel grid per region) were obtained from the authors;
they are not in the public release, which provides 16-bit CODEX stacks from another export and no H&E
(https://magdalenamat.github.io/Spatial_MAC_Map/, `Analysis_code_and_tab_data/CODEX/stacked_TIF_images/`). With the
authors' files:

```
$CELLPLEX_RAW/bc_crc/HE/{BC,CRC}_reg###.HE.tif            (H, W, 3) uint8
$CELLPLEX_RAW/bc_crc/IF_uint8/{BC,CRC}_reg###.IF.tif      (C, H, W) uint8; channels: panels/bc_crc_channels_{BC,CRC}.csv
```

```bash
python scripts/preprocess/bc_crc.py prep
sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/bc_crc/cellvit_filelist.csv $CELLPLEX_DATA/bc_crc/seg
python scripts/preprocess/bc_crc.py quant
```

46 regions (BC 16, CRC 30); excluded by QC: CRC_reg016 (0.755). Split by region, stratified by tissue.

## HNSCC-LN (`hnscc_ln`)

`urls/hnscc_ln.txt` (discovery cohort: 390 H&E, 390 CODEX cores and 6 channel-name files, ~247 GB, plus the clinical
table; Globus is much faster than HTTP: collection `47772002-3e5b-4fd3-b97c-18cee38d6df2`, path
`/biostudies/fire/S-BIAD/638/S-BIAD1638/Files/Submission2/HNSCC/`):

```
$CELLPLEX_RAW/hnscc_ln/HandE/{tma}_S{k}reg###_HandE.tif
$CELLPLEX_RAW/hnscc_ln/raw_TIFF/{tma}_S{k}reg###.tif, {tma}_S{k}_channelnames.txt
$CELLPLEX_RAW/hnscc_ln/metadata/HNSCCdiscovery_clinmetadata.csv
```

```bash
python scripts/preprocess/hnscc_ln.py prep
sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/hnscc_ln/cellvit_filelist.csv $CELLPLEX_DATA/hnscc_ln/seg
python scripts/preprocess/hnscc_ln.py quant
```

Excluded: 521_S1reg062 (QC 0.776), 523_S1reg001 (no clinical record). Unit = patient (CaseID), folds stratified by
nodal status.

## Bone marrow (`bone_marrow`)

`urls/bone_marrow.txt` (URL, file name; 12 H&E and 12 CODEX .qptiff, ~70 GB):

```
$CELLPLEX_RAW/bone_marrow/HE/{case}_HE.qptiff
$CELLPLEX_RAW/bone_marrow/IF/{case}_Scan1.qptiff
$CELLPLEX_RAW/bone_marrow/aligned/{case}.IF.ome.tiff       optional, see below
```

```bash
python scripts/preprocess/bone_marrow.py he
sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/bone_marrow/cellvit_filelist.csv $CELLPLEX_DATA/bone_marrow/seg
python scripts/preprocess/bone_marrow.py quant --workers 16                       # with aligned/
python scripts/preprocess/bone_marrow.py register && \
python scripts/preprocess/bone_marrow.py quant --workers 16 --source registered    # without aligned/
```

The truth of CellPlex was read from CODEX images registered to the H&E pixel grid that were provided with the data
(`aligned/`); they are not part of the public release. Without them, `register` applies the registration used for the
glioma cohort to the public CODEX; we have not compared that path with the images used. Excluded by QC: AML1_191
(0.595). Split by section (one patient each).

## Glioma (`glioma`)

`urls/glioma.txt` (37 sections x (CODEX, H&E) OME-TIFF, ~920 GB, plus `TableS1_SampleMetadata_v13.xlsx`; Globus
collection `47772002-3e5b-4fd3-b97c-18cee38d6df2`, path `/biostudies/fire/S-BIAD/840/S-BIAD2840/`):

```
$CELLPLEX_RAW/glioma/scans_IDHm/IDHm_exp*/{id}.ome.tif, {id}_HE.ome.tif
$CELLPLEX_RAW/glioma/scans_GBM/GBM_exp*/{id}.ome.tif, {id}_HE.ome.tif
$CELLPLEX_RAW/glioma/TableS1_SampleMetadata_v13.xlsx
```

```bash
python scripts/preprocess/glioma.py prep
python scripts/preprocess/glioma.py register                   # IDHA19, whose tissue fills the field: --coarse intensity
CELLVIT_ALL_PATCHES=1 sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/glioma/cellvit_filelist.csv $CELLPLEX_DATA/glioma/seg
python scripts/preprocess/glioma.py quant
```

Excluded by QC: IDHA19 (0.353). Unit = patient, folds stratified by tumour type; Orion-CRC and glioma are evaluated on
one fixed split (fold 0).

## Checks

Run on samples of the original data with the CellViT++ segmentations used for CellPlex, the ported steps reproduce
our outputs exactly: H&E arrays, cell regions, shifts, QC and truth for CRC-TMA (B_reg009), Multi-tumor (reg001),
BC/CRC (BC_reg001) and HNSCC-LN (521_S1reg001); the quantification of bone marrow (H33, from `aligned/`) and glioma
(IDHO03), and the glioma registration (IDHO03); the cells, tiles and folds written by `build_dataset.py` (Multi-tumor
reg001, CRC-TMA B_reg009, Orion-CRC CRC03_Scan1 from its quantification); and the pixel targets of Multi-tumor reg001
and CRC-TMA B_reg009. Not run here: CellViT++ itself, the Orion steps on the full public images (only their pixel
content was compared, above), and the bone-marrow registration path.
