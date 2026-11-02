"""CellViT++ detect_cells.py with process counts sized to the job's CPU allocation (same CLI as detect_cells.py).

CellViT++ (https://github.com/TIO-IKIM/CellViT-Plus-Plus, used at commit 463c5c4) sizes Ray (os.cpu_count() - 2
workers) and its patch DataLoader from the node's core count rather than the job's, which on large nodes means ~100
prestarted Ray workers and tens of GB of memory. Here, without editing the CellViT++ repository, Ray gets only the CPUs
its post-processing actors declare, worker prestart is off, the object store is capped (CELLVIT_OBJ_STORE_GB, it
spills to disk) and the DataLoader uses the allocated cores. Model, tiling, post-processing and outputs are unchanged.

CELLVIT_ALL_PATCHES=1 keeps every patch of the slide (no Otsu tissue prefilter; used for the pale glioma H&E, where
the prefilter drops most of the tissue; background patches simply yield no nuclei).

Setup: clone CellViT++, create its environment, download CellViT-SAM-H-x40-AMP.pth into its checkpoints/ and set
CELLVIT_DIR to the clone. Every cohort is segmented with this model at 0.25 um/px (magnification 40):

  python scripts/preprocess/run_cellvit.py --model $CELLVIT_DIR/checkpoints/CellViT-SAM-H-x40-AMP.pth \
      --outdir $CELLPLEX_DATA/multitumor/seg --batch_size 8 \
      process_dataset --filelist $CELLPLEX_DATA/multitumor/cellvit_filelist.csv --wsi_extension tif
  python scripts/preprocess/run_cellvit.py --model ... --outdir $CELLPLEX_DATA/orion_crc/seg_raw/CRC03_Scan1 \
      process_wsi --wsi_path $CELLPLEX_DATA/orion_crc/wsi_cv/CRC03_Scan1.tif \
      --wsi_properties '{"slide_mpp": 0.25, "magnification": 40}'
"""
import os
import sys

os.environ.setdefault('RAY_enable_worker_prestart', '0')
os.environ.setdefault('RAY_prestart_worker_first_driver', '0')

CV = os.environ.get('CELLVIT_DIR', 'CellViT-Plus-Plus')
sys.path.insert(0, CV)
sys.path.insert(0, f'{CV}/cellvit')

import numpy as np  # noqa: E402
import ray  # noqa: E402
from cellvit.inference import inference_disk  # noqa: E402

OBJ_STORE_GB = float(os.environ.get('CELLVIT_OBJ_STORE_GB', '8'))


def _setup_worker(self):
    ncpu = len(os.sched_getaffinity(0))
    self.ray_actors = int(np.clip(self.batch_size / 2, 4, 8))
    ray.init(num_cpus=8 * self.ray_actors, include_dashboard=False,
             object_store_memory=int(OBJ_STORE_GB * 1e9),
             _temp_dir=os.environ.get('RAY_TMPDIR'),
             runtime_env={'env_vars': {'PYTHONPATH': inference_disk.project_root}})
    self.num_workers = int(np.clip(ncpu - 2, 1, 4 * self.batch_size))
    self.logger.info(f'lean setup: {ncpu} allocated cpus, {self.num_workers} loader workers, '
                     f'{self.ray_actors} ray actors, object store {OBJ_STORE_GB} GB')


inference_disk.CellViTInference._setup_worker = _setup_worker

if os.environ.get('CELLVIT_ALL_PATCHES') == '1':
    from cellvit.inference import inference_memory as _im
    _Cfg = _im.LivePatchWSIConfig

    def _all_patches(**kw):
        kw.update(min_intersection_ratio=0.0, apply_prefilter=False)
        return _Cfg(**kw)
    _im.LivePatchWSIConfig = _all_patches

if __name__ == '__main__':
    os.chdir(CV)
    from cellvit import detect_cells
    detect_cells.main()
