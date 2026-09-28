"""Polyline helpers for rivers and streams: corner-cutting smoothing and meanders."""
import numpy as np


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def chaikin(p, iterations=3):
    """Chaikin corner cutting; every column (x, y and any per-point values) is smoothed alike."""
    p = np.asarray(p, np.float64)
    for _ in range(iterations):
        if len(p) < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), p.shape[1]))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[:1], mid, p[-1:]])
    return p


def meander(line, slope, dx, strength, seed, width_col=2):
    """Wind a line sideways where the land is flat (meanders), not in steep valleys.
    `line` rows start with x, y (metres); column `width_col` is the channel width."""
    if strength <= 0 or len(line) < 4:
        return line
    xy = line[:, :2]
    seg = np.diff(xy, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    tang = np.gradient(xy, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], -1)
    n = slope.shape[0]
    ij = np.clip((xy / dx).astype(int), 0, n - 1)
    flat = 1.0 - _smoothstep(slope[ij[:, 1], ij[:, 0]] / 0.03)
    width = line[:, width_col]
    rng = np.random.default_rng(seed)
    wave = np.sin(s / (12 * width + 60) * 2 * np.pi + rng.uniform(0, 6.3)) * 0.6 \
        + np.sin(s / (5 * width + 25) * 2 * np.pi + rng.uniform(0, 6.3)) * 0.4
    off = strength * 3.0 * width * flat * wave
    off[0] = off[-1] = 0.0
    out = line.copy()
    out[:, :2] = xy + nrm * off[:, None]
    return out
