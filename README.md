# CellPlex

CellPlex is a benchmark for predicting single-cell spatial proteomics from H&E. It assembles seven cohorts in which
H&E and multiplexed immunofluorescence were acquired on the same tissue section, processes them with one pipeline
into per-cell protein targets, and scores every model on the same cells, splits and metrics.

* **Task**: from an H&E image and the nuclei segmented on it, predict the level of every protein of the cohort's panel
  in every cell (log1p of the mean intensity over the nucleus and a 2 µm rim).
* **Cohorts**: 717 samples, 44.8 million cells and 149 proteins, from colorectal and breast cancer to lymph-node
  metastases, bone marrow and glioma, on two imaging platforms (CODEX and Orion).
* **Models**: pathology foundation models (UNI2-h, Virchow2, H-optimus-0) adapted to cells with
  [CELLO](https://github.com/zjgao02/CELLO), linear probes on their frozen features, and the virtual-staining models Eva
  and GigaTIME, as released and fine-tuned on each cohort.
* **Metrics**: per test sample, the Pearson correlation (PCC) and the cell-level structural similarity (SSIM) between
  predicted and measured levels across the sample's cells, averaged over proteins and then over test samples.

This repository holds the code to build the benchmark from the original data releases, train and test every model,
and reproduce the tables. The processed data are not redistributed; `data/README.md` lists where each cohort is
downloaded from. Trained models of every fold are on Hugging Face:
[gaozijun/CellPlex](https://huggingface.co/gaozijun/CellPlex).

## Quick start

```bash
conda create -n cellplex python=3.10 -y && conda activate cellplex
conda install -c conda-forge openslide -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # for your CUDA version
pip install -r requirements.txt
huggingface-cli login            # access to UNI2-h, Virchow2 and H-optimus-0 is requested on their model pages

export CELLPLEX_RAW=/path/to/raw          # original releases, one folder per cohort (data/README.md)
export CELLPLEX_DATA=/path/to/cellplex    # processed cohorts
export CELLPLEX_RUNS=/path/to/runs        # checkpoints and test predictions

python scripts/preprocess/build_dataset.py --cohort multitumor          # after the cohort's preprocessing steps
python scripts/train.py --cohort multitumor --fold 0 --encoder hoptimus0
python scripts/evaluate.py --cohorts multitumor
```

## Cohorts

| Cohort | Tissue | Platform | Format | µm/px | Samples | Cells | Proteins | Split unit | Folds |
|---|---|---|---|---|---|---|---|---|---|
| `orion_crc` | Colorectal cancer | Orion | WSI | 0.325 | 41 | 38.0 M | 16 | patient | fixed |
| `crc_tma` | Colorectal cancer | CODEX | TMA | 0.377 | 130 | 234.5 K | 56 | core | 5 |
| `bc_crc` | Breast, colorectal cancer | CODEX | regions | 0.507 | 45 | 516.9 K | 36 | region | 5 |
| `multitumor` | Multiple tumor types | CODEX | TMA | 0.377 | 66 | 147.9 K | 57 | core | 5 |
| `hnscc_ln` | HNSCC lymph nodes | CODEX | TMA | 0.377 | 388 | 1.47 M | 57 | patient | 5 |
| `bone_marrow` | Bone marrow | CODEX | WSI | 0.249 | 11 | 772.4 K | 52 | patient | 5 |
| `glioma` | Glioma | CODEX | WSI | 0.249 | 36 | 3.59 M | 68 | patient | fixed |

`data/splits/{cohort}.csv` assigns every split unit to a fold. For fold k, the units of fold k are the test set, the
first half of the units of fold k + 1 the validation set (checkpoint selection only), and the rest the training set.
Orion-CRC and glioma, the cohorts with the largest tissue area, use fold 0 as one fixed split.
`data/panels/{cohort}.csv` lists each panel's markers in output order with their canonical protein names, by which
proteins are matched across cohorts.

## Data

`data/README.md` gives, for each cohort, the original publication and repository, its licence, the files to download,
the expected layout under `$CELLPLEX_RAW/{cohort}/` and the processing commands. In short:

1. **H&E**: brought to the cohort's native resolution (stitched into a pyramidal TIFF for whole slides) and resampled
   to 0.25 µm/px for segmentation.
2. **Nuclei**: segmented on the H&E with [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) (SAM-H, x40),
   so that every target belongs to a cell visible at inference time (`scripts/preprocess/run_cellvit.py`).
3. **Alignment**: the cell regions are moved into the multiplex frame, by an integer shift per core or region for the
   arrays and regions (imaged on one pixel grid with the H&E) and by a registration for the glioma slides; a sample is
   excluded when its nuclear stain is not brighter in the nucleus than in the ring around it for at least 80% of cells.
4. **Targets**: per cell and protein, log1p of the mean intensity over the nucleus and a 2 µm rim.
5. **Benchmark format** (`scripts/preprocess/build_dataset.py`): `$CELLPLEX_DATA/{cohort}/` with `meta.json`,
   `cells/{sample}.npz` (centroids and targets), `tiles.csv` (non-overlapping 224-pixel tiles at native resolution;
   a cell belongs to the tile holding its centroid) and `folds.csv`.

`scripts/preprocess/pixel_targets.py` additionally writes the pixel-level multiplex targets used to fine-tune the
specialist models.

The H&E of BC/CRC and the bone-marrow multiplex images registered to the H&E were obtained from the authors and are
not part of the public releases; `data/README.md` describes what can be rebuilt from public files for these cohorts.

## Pretrained models

[gaozijun/CellPlex](https://huggingface.co/gaozijun/CellPlex) holds the model of every fold of every cohort: CELLO
with each foundation model (`cello/{encoder}/{cohort}/fold{k}`), the linear probes (`linear_probe/...`) and the
fine-tuned Eva and GigaTIME (`eva_ft/...`, `gigatime_ft/...`), with inference code for whole images and nucleus
centroids. Access is granted on request.

## Training

**CELLO** (`scripts/train.py`, `configs/cello.yaml`): one forward pass of the foundation model per 224-pixel tile, the
token map bilinearly sampled at each nucleus, refined by a distance-decay cross-attention over the tile and fused with
the CLS token, then decoded into the panel. Huber loss on the cells plus 0.5 × a tile-level loss on the summed levels;
the encoder is frozen for the first 10% of training, its learning rate warmed up over the next 30% and it is then
fine-tuned with the rest of the model. One GPU per run.

```bash
python scripts/train.py --cohort multitumor --fold 0 --encoder hoptimus0          # uni2 | virchow2 | hoptimus0
python scripts/train.py --cohort multitumor --fold 0 --encoder hoptimus0 --eval_only
```

Epochs: 200 for the array and region cohorts, 5 for Orion-CRC and glioma, 50 for bone marrow. The checkpoint with the
best validation score (mean over validation samples of the sample's mean protein PCC) is tested on every cell of the
test units. Output: `$CELLPLEX_RUNS/{cohort}/{encoder}/fold{k}/`.

**Linear probe** (`scripts/linear_probe.py`): the frozen encoder's final token map sampled at every nucleus, and a
ridge regression to all proteins whose regularisation is chosen on the validation set.

```bash
python scripts/linear_probe.py extract --cohort multitumor --encoder hoptimus0     # GPU, once per cohort and encoder
python scripts/linear_probe.py fit --cohort multitumor --encoder hoptimus0         # CPU, every fold
```

**Specialist models** (`scripts/finetune_specialist.py`, `configs/specialists.yaml`): Eva and GigaTIME as released
(`--zeroshot`) and fine-tuned on each fold with their own pixel-level objectives and a training budget matched to
CELLO. Their code comes from their own repositories; see `docs/specialists.md`.

```bash
python scripts/finetune_specialist.py --model eva --cohort multitumor --fold 0
python scripts/finetune_specialist.py --model gigatime --cohort multitumor --fold 0 --zeroshot
```

## Evaluation

```bash
python scripts/evaluate.py                       # every cohort and method with test outputs
```

`scripts/evaluate.py` scores the test predictions of every run: PCC, Spearman and SSIM per sample and protein, the
cohort scores (mean ± standard error over test samples) and per-protein tables, written to `$CELLPLEX_RUNS/results/`.

**Transfer across cohorts** (`scripts/cross_cohort.py`): the fold-0 model of one cohort is applied without further
training to the test samples of every other cohort, with the H&E resampled to the training cohort's resolution, and
scored on the proteins the two panels share.

```bash
python scripts/cross_cohort.py run --src crc_tma --encoder hoptimus0
python scripts/cross_cohort.py summary
```

## Inference on new images

```bash
python scripts/predict.py --ckpt $CELLPLEX_RUNS/multitumor/hoptimus0/fold0/best_model.pt --image slide.svs --mpp 0.25 \
    --cells centroids.csv --out predictions.parquet
```

`centroids.csv` holds the nucleus centroids (`x, y`, level-0 pixels); the image is resampled to the resolution of the
model's training cohort. The Hugging Face repository has the same inference for its released models.

## Repository layout

```
cellplex/             model (CELLO), data loading, metrics, cohort registry, paths, specialist wrappers
cellplex/preprocess/  shared preprocessing helpers
configs/              training recipes of CELLO and the specialist models
data/                 data sources (README.md), download lists (urls/), splits and panels
docs/                 setup of the specialist models
scripts/preprocess/   raw data -> benchmark format, per cohort
scripts/              training, linear probe, specialists, evaluation, transfer, inference
```

## Licence

Each cohort is used under the licence of its original release (`data/README.md`); the processed data are not
redistributed. The foundation models and specialist models are obtained from their own repositories under their
licences and terms of use (UNI2-h and Virchow2: CC BY-NC-ND 4.0; H-optimus-0 and GigaTIME: Apache 2.0; Eva: see its
repository).

## Acknowledgements

CellPlex builds on the cohorts released by Lin et al. (Orion-CRC), Schürch et al. (CRC-TMA and Multi-tumor), Matusiak
et al. (BC/CRC), Haist et al. (HNSCC-LN), Bandyopadhyay et al. (bone marrow) and Hoefflin et al. (glioma); on the
foundation models UNI2-h, Virchow2 and H-optimus-0; on CellViT++ for nucleus segmentation; on CELLO; and on the Eva
and GigaTIME models and code.
