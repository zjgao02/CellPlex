"""Where CellPlex reads raw downloads and writes processed cohorts and runs (set by environment variables).

    CELLPLEX_RAW    original releases, one folder per cohort as laid out in data/README.md   (default ./raw)
    CELLPLEX_DATA   processed cohorts written by scripts/preprocess/build_dataset.py        (default ./cellplex_data)
    CELLPLEX_RUNS   checkpoints and test predictions                                         (default ./runs)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.abspath(os.environ.get('CELLPLEX_RAW', os.path.join(REPO, 'raw')))
DATA = os.path.abspath(os.environ.get('CELLPLEX_DATA', os.path.join(REPO, 'cellplex_data')))
RUNS = os.path.abspath(os.environ.get('CELLPLEX_RUNS', os.path.join(REPO, 'runs')))
SPLITS = os.path.join(REPO, 'data', 'splits')
PANELS = os.path.join(REPO, 'data', 'panels')


def raw_dir(cohort):
    return os.path.join(RAW, cohort)


def cohort_dir(cohort):
    return os.path.join(DATA, cohort)


def run_dir(cohort, method, fold):
    return os.path.join(RUNS, cohort, method, f'fold{fold}')
