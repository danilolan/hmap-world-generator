"""Colour maps shared by the stages' views, and PNG encoding."""
import base64
import io

import numpy as np
from PIL import Image


def _ramp(values, stops):
    xs = np.array([s[0] for s in stops], float)
    cs = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(values, xs, cs[:, k]) for k in range(3)], -1)


def gray(a, lo=None, hi=None):
    lo = np.nanmin(a) if lo is None else lo
    hi = np.nanmax(a) if hi is None else hi
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.repeat((t * 255)[..., None], 3, -1).astype(np.uint8)


def hillshade(height_m, cell_m, azimuth=315.0, altitude=40.0):
    gy, gx = np.gradient(height_m, cell_m)
    az, alt = np.radians(azimuth), np.radians(altitude)
    nl = np.sqrt(gx ** 2 + gy ** 2 + 1)
    return np.clip((np.cos(alt) * np.cos(az) * -gx + np.cos(alt) * np.sin(az) * gy + np.sin(alt)) / nl, 0, 1)


TERRAIN_STOPS = [(-1000, (15, 30, 75)), (-200, (30, 70, 135)), (-1, (70, 130, 180)), (0, (95, 150, 75)),
                 (300, (135, 170, 85)), (900, (175, 165, 105)), (1800, (145, 120, 95)),
                 (2800, (165, 160, 155)), (3600, (245, 245, 250))]


def terrain(height_m, cell_m, shade=True):
    """Hypsometric tints with hillshade; water below 0 m."""
    col = _ramp(height_m, TERRAIN_STOPS)
    if shade:
        s = hillshade(height_m, cell_m)[..., None]
        land = (height_m >= 0)[..., None]
        col = np.where(land, col * (0.35 + 0.75 * s), col * (0.85 + 0.15 * s))
    return np.clip(col, 0, 255).astype(np.uint8)


def heat(a, lo, hi):
    """Blue (cold) to red (hot)."""
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return _ramp(t, [(0, (40, 60, 160)), (0.35, (90, 180, 220)), (0.5, (240, 240, 200)),
                     (0.7, (245, 170, 70)), (1, (180, 30, 30))]).astype(np.uint8)


def moisture(a, lo=0.0, hi=1.0):
    """Dry brown to wet blue-green."""
    t = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return _ramp(t, [(0, (170, 120, 70)), (0.4, (215, 205, 120)), (0.7, (90, 170, 90)),
                     (1, (30, 90, 160))]).astype(np.uint8)


def categorical(labels, palette):
    """labels: int array; palette: list of RGB tuples indexed by label."""
    pal = np.array(palette, np.uint8)
    return pal[np.clip(labels, 0, len(pal) - 1)]


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def data_url(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(rgb)).decode()
