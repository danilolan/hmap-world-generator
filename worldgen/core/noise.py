"""Seeded fractal noise on a square grid.

Each octave is a coarse grid of Gaussian values upsampled with cubic
interpolation. The coarse grids depend only on the seed and the octave, so a
seed gives the same pattern at any preview resolution.
"""
import numpy as np
from scipy import ndimage


def fbm(n: int, octaves: int, base_cells: float, seed: int, persistence: float = 0.5, lacunarity: float = 2.0):
    rng = np.random.default_rng(seed)
    acc = np.zeros((n, n), np.float32)
    amp, total, cells = 1.0, 0.0, float(base_cells)
    for _ in range(octaves):
        c = max(1, int(round(cells)))
        grid = rng.standard_normal((c + 3, c + 3)).astype(np.float32)
        acc += amp * ndimage.zoom(grid, n / c, order=3)[:n, :n]
        total += amp
        amp *= persistence
        cells *= lacunarity
    acc /= total
    return acc / (acc.std() + 1e-6)


def warp(field: np.ndarray, strength_cells: float, seed: int, base_cells: float = 4):
    """Domain warping: sample `field` at positions displaced by two noise fields."""
    n = field.shape[0]
    wx = fbm(n, 3, base_cells, seed + 1) * strength_cells
    wy = fbm(n, 3, base_cells, seed + 2) * strength_cells
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return ndimage.map_coordinates(field, [np.clip(yy + wy, 0, n - 1), np.clip(xx + wx, 0, n - 1)],
                                   order=1, mode="nearest")
