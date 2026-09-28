"""Noise evaluated at absolute world coordinates (metres), point by point.

Used for everything finer than the preview grid (micro relief): the value at a
place depends only on its coordinates and the seed, so the detail window, any
preview resolution and the 2 m export all see exactly the same terrain.
Coordinates follow the tool's image axes: x = metres east from the west edge,
y = metres south from the north edge.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True, inline="always")
def _hash(ix, iy, seed):
    h = np.int64(ix) * 374761393 + np.int64(iy) * 668265263 + np.int64(seed) * 982451653
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 8388607.5 - 1.0


@nb.njit(cache=True, inline="always")
def _grad(ix, iy, seed, dx, dy):
    """Dot product of the offset with a pseudo-random unit gradient at a lattice point."""
    a = _hash(ix, iy, seed) * np.pi
    return np.cos(a) * dx + np.sin(a) * dy


@nb.njit(cache=True, inline="always")
def _value(x, y, seed):
    """2D gradient (Perlin) noise. Value noise, used before, left square-ish,
    lattice-aligned shapes that showed at high resolution."""
    ix = np.floor(x)
    iy = np.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    n00 = _grad(ix, iy, seed, fx, fy)
    n10 = _grad(ix + 1, iy, seed, fx - 1, fy)
    n01 = _grad(ix, iy + 1, seed, fx, fy - 1)
    n11 = _grad(ix + 1, iy + 1, seed, fx - 1, fy - 1)
    top = n00 + (n10 - n00) * ux
    bot = n01 + (n11 - n01) * ux
    return 1.6 * (top + (bot - top) * uy)          # scaled to roughly [-1, 1]


@nb.njit(cache=True, parallel=True)
def _fbm(x, y, wavelength, octaves, seed, mode, gain):
    out = np.empty(x.shape[0], np.float32)
    for i in nb.prange(x.shape[0]):
        f = 1.0 / wavelength
        amp = 1.0
        tot = 0.0
        acc = 0.0
        for o in range(octaves):
            # rotate each octave a little to hide the lattice
            ca = np.cos(0.5 * o)
            sa = np.sin(0.5 * o)
            px = (x[i] * ca - y[i] * sa) * f
            py = (x[i] * sa + y[i] * ca) * f
            v = _value(px, py, seed + o * 101)
            if mode == 1:        # ridged: sharp crests
                v = 1.0 - 2.0 * abs(v)
            elif mode == 2:      # billow: rounded domes, creased valleys
                v = 2.0 * abs(v) - 1.0
            acc += amp * v
            tot += amp
            amp *= gain
            f *= 2.0
        out[i] = acc / tot
    return out


MODES = {"fbm": 0, "ridged": 1, "billow": 2}


FBM_STD = 0.23   # measured standard deviation of the "fbm" mode (values reach about +-0.75)


def fbm_at(x_m, y_m, wavelength_m, octaves, seed, mode="fbm", gain=0.5):
    """Fractal noise at world coordinates; x_m and y_m are arrays of the same shape.
    "fbm" mode: mean 0, standard deviation about FBM_STD (use fbm_unit for std 1).
    `wavelength_m` is the size of the largest features."""
    x = np.ascontiguousarray(x_m, np.float64).ravel()
    y = np.ascontiguousarray(y_m, np.float64).ravel()
    return _fbm(x, y, float(wavelength_m), int(octaves), int(seed) % 2_000_000_000, MODES[mode], float(gain)).reshape(np.shape(x_m))


def fbm_unit(x_m, y_m, wavelength_m, octaves, seed, gain=0.5):
    """fbm_at scaled to a standard deviation of about 1."""
    return fbm_at(x_m, y_m, wavelength_m, octaves, seed, "fbm", gain) / FBM_STD
