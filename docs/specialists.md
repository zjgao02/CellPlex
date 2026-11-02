# Specialist models: Eva and GigaTIME

`scripts/finetune_specialist.py` evaluates two H&E-to-multiplex models as released (`--zeroshot`) and after
fine-tuning on a cohort with their own pixel-level objectives, on the same folds, cells, targets and metrics as the
CELLO runs. Their code is not part of this repository; install it from the original repositories.

## Eva

```bash
git clone https://github.com/YAndrewL/Eva.git
pip install omegaconf einops            # Eva's dependencies beyond requirements.txt (see its env.yaml)
mkdir -p Eva/marker_embeddings          # GenePT embeddings, as described in Eva's README:
# download GenePT_gene_protein_embedding_model_3_text.pickle from https://zenodo.org/records/10833191
# and save it as Eva/marker_embeddings/GenePT_embedding.pkl
export EVA_REPO=$PWD/Eva                # EVA_WORKDIR (default: EVA_REPO) is the directory holding marker_embeddings/
```

The released weights (`Eva_ft.ckpt` of [yandrewl/Eva](https://huggingface.co/yandrewl/Eva)) are downloaded from the
Hugging Face hub; set `EVA_CKPT` to use a local copy.

## GigaTIME

```bash
git clone https://github.com/prov-gigatime/GigaTIME.git
export GIGATIME_REPO=$PWD/GigaTIME      # the model definition is GigaTIME/scripts/archs.py
```

The released weights (`model.pth` of [prov-gigatime/GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME); request
access on the model page) are downloaded from the Hugging Face hub; set `GIGATIME_CKPT` to use a local copy.

## Inputs

Fine-tuning needs pixel-level multiplex targets in the H&E frame, and testing needs the cell regions to average the
predicted maps over; both are written by the preprocessing scripts (`scripts/preprocess/`). The files are listed in
`cellplex/specialists/data.py`.

## Running

```bash
python scripts/finetune_specialist.py --model eva --cohort multitumor --fold 0 --epochs N    # fine-tune and test
python scripts/finetune_specialist.py --model gigatime --cohort multitumor --fold 0 --zeroshot
python scripts/finetune_specialist.py --model eva --cohort multitumor --fold 0 --eval_only   # test best_model.pt again
```

Batch sizes and learning rates are in `configs/specialists.yaml`; the training length is set with `--epochs`. Eva is trained with a
micro-batch of 8 and 4 accumulation steps (a 96 GB GPU); lower `--bs` and raise `--accum` on smaller GPUs.
Outputs follow the CELLO runs (`$CELLPLEX_RUNS/{cohort}/{eva,gigatime}_{ft,zeroshot}/fold{k}/test/`), so
`scripts/evaluate.py` scores all methods together.
