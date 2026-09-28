"""Small raster helpers shared by stages: warped Voronoi, labels, drawing."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .noise import fbm


def warped_voronoi(n, points_cells, warp_cells, seed, warp_features=3.0):
    """Nearest-point labels on an n x n grid whose sampling coordinates are
    displaced by noise, so cell borders wander instead of being straight."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    if warp_cells > 0:
        xx = xx + fbm(n, 3, warp_features, seed + 1) * warp_cells
        yy = yy + fbm(n, 3, warp_features, seed + 2) * warp_cells
    tree = cKDTree(points_cells)
    _, idx = tree.query(np.stack([yy.ravel(), xx.ravel()], -1), workers=-1)
    return idx.reshape(n, n).astype(np.int32)


def adjacency(labels):
    """Set of (a, b) label pairs, a < b, that touch (4-neighbourhood)."""
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = a != b
        lo = np.minimum(a[m], b[m])
        hi = np.maximum(a[m], b[m])
        pairs.update(zip(lo.tolist(), hi.tolist()))
    return pairs


def boundary_pairs(labels):
    """Per pixel on a label border, the (lo, hi) pair it separates; -1 elsewhere.
    Returns two int arrays (lo, hi)."""
    lo = np.full(labels.shape, -1, np.int32)
    hi = np.full(labels.shape, -1, np.int32)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(labels, (-dy, -dx), (0, 1))
        m = nb != labels
        if dy == 1:
            m[-1, :] = False
        if dy == -1:
            m[0, :] = False
        if dx == 1:
            m[:, -1] = False
        if dx == -1:
            m[:, 0] = False
        lo[m] = np.minimum(labels[m], nb[m])
        hi[m] = np.maximum(labels[m], nb[m])
    return lo, hi


def component_areas(mask, cell_area_km2, structure8=True):
    st = np.ones((3, 3)) if structure8 else None
    lab, cnt = ndimage.label(mask, structure=st)
    areas = ndimage.sum(np.ones_like(lab, float), lab, index=np.arange(1, cnt + 1)) * cell_area_km2
    return lab, cnt, areas


def draw_line(img, y0, x0, y1, x1, color, width=1):
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    h, w = img.shape[:2]
    r = width // 2
    for y, x in zip(ys, xs):
        yi, xi = int(round(y)), int(round(x))
        img[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)] = color


def palette(count, seed, lo=60, hi=220):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, size=(max(count, 1), 3)).astype(np.uint8)
