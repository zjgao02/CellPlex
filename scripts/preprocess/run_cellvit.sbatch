#!/bin/bash
#SBATCH --job-name=cellvit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# CellViT++ (SAM-H x40) on every image of a cohort's file list (written by its prep step):
#   sbatch scripts/preprocess/run_cellvit.sbatch $CELLPLEX_DATA/multitumor/cellvit_filelist.csv $CELLPLEX_DATA/multitumor/seg
# Requires CELLVIT_DIR (CellViT++ clone with checkpoints/CellViT-SAM-H-x40-AMP.pth) and its Python environment active.
# For the glioma sections, export CELLVIT_ALL_PATCHES=1 before submitting. Whole slides with millions of nuclei need
# more memory (CellViT++ keeps every cell in memory); Orion slides are run one by one with process_wsi.
set -euo pipefail
FL=$(realpath "$1"); OUT=$(realpath -m "$2"); mkdir -p "$OUT"
export CELLVIT_OBJ_STORE_GB=${CELLVIT_OBJ_STORE_GB:-4} RAY_TMPDIR=${SLURM_TMPDIR:-/tmp}/ray_${SLURM_JOB_ID:-$$}
mkdir -p "$RAY_TMPDIR"
cd "$(dirname "$0")/../.."
python scripts/preprocess/run_cellvit.py --model "$CELLVIT_DIR/checkpoints/CellViT-SAM-H-x40-AMP.pth" --outdir "$OUT" \
    --batch_size 8 process_dataset --filelist "$FL" --wsi_extension tif
echo "$(ls "$OUT"/*_cells.json 2>/dev/null | wc -l) cells.json in $OUT"
