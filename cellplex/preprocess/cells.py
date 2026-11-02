"""Nucleus polygons of a whole slide (parquet: geometry WKB, cell_id, cx, cy, class) -> label maps."""
import cv2
import numpy as np
import pyarrow.parquet as pq
import shapely


class Cells:
    """All nuclei of one slide, flattened for fast rasterisation.

    xy (V, 2) float32 polygon vertices, ptr (N+1,) offsets into xy, bb (N, 4) xmin ymin xmax ymax, cx / cy centroids,
    cid the parquet cell_id, cls the class.
    """

    def __init__(self, path):
        t = pq.read_table(path, columns=['geometry', 'cell_id', 'cx', 'cy', 'class'])
        geom = shapely.from_wkb(t.column('geometry').to_numpy(zero_copy_only=False))
        xy, gid = shapely.get_coordinates(geom, return_index=True)
        self.n = len(geom)
        self.ptr = np.r_[0, np.cumsum(np.bincount(gid, minlength=self.n))].astype(np.int64)
        self.xy = xy.astype(np.float32)
        self.bb = shapely.bounds(geom).astype(np.float32)
        self.cid = t.column('cell_id').to_numpy()
        self.cx = t.column('cx').to_numpy().astype(np.float32)
        self.cy = t.column('cy').to_numpy().astype(np.float32)
        self.cls = np.asarray(t.column('class').to_pandas().astype(str))

    def draw(self, idx, x0, y0, shape, scale=1.0):
        """Label map (int32) of nuclei `idx` drawn as 1..len(idx); pixel (r, c) covers x0 + c / scale, y0 + r / scale."""
        lab = np.zeros(shape, np.int32)
        for j, c in enumerate(idx):
            pts = (self.xy[self.ptr[c]:self.ptr[c + 1]] - (x0, y0)) * scale
            cv2.fillPoly(lab, [np.round(pts).astype(np.int32)], int(j + 1))
        return lab


def tile_index(bb, tile, margin, ny, nx):
    """{(iy, ix): cell indices} of every grid tile each cell's bbox (+margin) touches (a nucleus spans <= 2 x 2 tiles)."""
    ix0 = np.floor((bb[:, 0] - margin) / tile).astype(np.int64)
    ix1 = np.floor((bb[:, 2] + margin) / tile).astype(np.int64)
    iy0 = np.floor((bb[:, 1] - margin) / tile).astype(np.int64)
    iy1 = np.floor((bb[:, 3] + margin) / tile).astype(np.int64)
    assert ((ix1 - ix0) <= 1).all() and ((iy1 - iy0) <= 1).all()
    keys, ids = [], []
    n = np.arange(len(bb))
    for iy, ix in ((iy0, ix0), (iy0, ix1), (iy1, ix0), (iy1, ix1)):
        keys.append(iy * nx + ix); ids.append(n)
    keys, ids = np.concatenate(keys), np.concatenate(ids)
    pair = np.unique(np.stack([keys, ids], 1), axis=0)          # drop duplicates (ix0 == ix1 ...)
    ok = (pair[:, 0] >= 0) & (pair[:, 0] < ny * nx)
    pair = pair[ok]
    uk, start = np.unique(pair[:, 0], return_index=True)
    groups = np.split(pair[:, 1], start[1:])
    return {divmod(int(k), nx): g for k, g in zip(uk, groups)}
