"""Carving line features (river channels) into a regular window of heights."""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def carve_segments(ny, nx, x0, y0, step, segs, bed):
    """Depth to cut at each cell of a window whose cell (r, c) sits at world
    (x0 + c*step, y0 + r*step). segs rows: ax, ay, bx, by, half width, depth.
    Rounded cross-section reaching 1.6 half widths; the deepest cut wins; cells
    inside the half width are marked in `bed`."""
    cut = np.zeros((ny, nx))
    for s in range(segs.shape[0]):
        ax, ay, bx, by, hw, depth = segs[s]
        reach = hw * 1.6
        c0 = int((min(ax, bx) - reach - x0) / step) - 1
        c1 = int((max(ax, bx) + reach - x0) / step) + 2
        r0 = int((min(ay, by) - reach - y0) / step) - 1
        r1 = int((max(ay, by) + reach - y0) / step) + 2
        if c1 < 0 or r1 < 0 or c0 >= nx or r0 >= ny:
            continue
        dx = bx - ax
        dy = by - ay
        ll = dx * dx + dy * dy + 1e-9
        for r in range(max(r0, 0), min(r1, ny)):
            py = y0 + r * step
            for c in range(max(c0, 0), min(c1, nx)):
                px = x0 + c * step
                t = ((px - ax) * dx + (py - ay) * dy) / ll
                t = min(max(t, 0.0), 1.0)
                qx = ax + t * dx - px
                qy = ay + t * dy - py
                d = np.sqrt(qx * qx + qy * qy)
                if d < reach:
                    u = d / reach
                    v = depth * (1.0 - u * u)
                    if v > cut[r, c]:
                        cut[r, c] = v
                    if d < hw:
                        bed[r, c] = True
    return cut


BANK_RISE = 0.5          # river banks rise 0.5 m per metre from the water (about 27 degrees)


@nb.njit(cache=True)
def carve_channels(h, base, x0, y0, step, segs, water, flood_ratio):
    """Carve streams and rivers into a detail window. segs rows: ax, ay, bx, by, half
    width, depth, water surface at a, surface at b, reach of the valley floor, head
    (1 = a line's first segment: nothing is carved upslope of its start, where the
    round cap would otherwise notch the hillside above a spring). The
    surface lies below the ground under the line by the banks' height (see the hydrology
    stage), and carving only lowers the ground. The bed is the one the recurring flood
    cuts (flood_ratio times the mean flow, hydraulic geometry: width ~ Q^0.5, depth ~
    Q^0.35): inside the mean flow's half width a rounded trough under the water surface,
    1.5 times the mean depth at its middle (marked in `water`); out to the flood's half
    width a low bench rising to the flood's level (bars and banks the floods cover);
    beyond it a bank rising BANK_RISE per metre, down to which the ground is cut, fading
    out toward the reach, which opens a valley floor where the
    channel crosses a bump and a small V valley where it runs beside higher ground (the
    bank once stopped at `base`, the macro ground, which left sheer walls of several
    metres wherever the line ran beside the grid's valley; `base` is kept in the
    signature, unused). The nearest channel, in units of each channel's own reach, rules
    each cell."""
    ny, nx = h.shape
    wide = np.sqrt(flood_ratio)            # flood width over mean width (w ~ Q^0.5)
    deep = flood_ratio ** 0.35 - 1.0       # flood depth above the mean water, over the mean depth
    best = np.full((ny, nx), 1e30)
    surf = np.zeros((ny, nx))
    hw_at = np.zeros((ny, nx))
    dep_at = np.zeros((ny, nx))
    d_at = np.zeros((ny, nx))
    for s in range(segs.shape[0]):
        ax, ay, bx, by, hw, depth, sa, sb, cor, head = segs[s]
        c0 = int((min(ax, bx) - cor - x0) / step) - 1
        c1 = int((max(ax, bx) + cor - x0) / step) + 2
        r0 = int((min(ay, by) - cor - y0) / step) - 1
        r1 = int((max(ay, by) + cor - y0) / step) + 2
        if c1 < 0 or r1 < 0 or c0 >= nx or r0 >= ny:
            continue
        dx = bx - ax
        dy = by - ay
        ll = dx * dx + dy * dy + 1e-9
        for r in range(max(r0, 0), min(r1, ny)):
            py = y0 + r * step
            for c in range(max(c0, 0), min(c1, nx)):
                px = x0 + c * step
                t = ((px - ax) * dx + (py - ay) * dy) / ll
                if head > 0.5 and t < 0.0:
                    continue
                t = min(max(t, 0.0), 1.0)
                qx = ax + t * dx - px
                qy = ay + t * dy - py
                d = np.sqrt(qx * qx + qy * qy)
                u = d / cor
                if u < 1.0 and u < best[r, c]:
                    best[r, c] = u
                    surf[r, c] = sa + t * (sb - sa)
                    hw_at[r, c] = hw
                    dep_at[r, c] = depth
                    d_at[r, c] = d
    for r in range(ny):
        for c in range(nx):
            u = best[r, c]
            if u >= 1.0:
                continue
            d = d_at[r, c]
            hw = hw_at[r, c]
            s = surf[r, c]
            g = h[r, c]
            if d < hw:
                v = d / hw
                g = min(g, s - max(1.5 * dep_at[r, c], 0.3) * (1.0 - v * v))
                water[r, c] = True
            elif d < 0.5 * step:
                # a channel narrower than the window's cells still shows as one cell of water
                g = min(g, s - 0.05)
                water[r, c] = True
            else:
                hwb = hw * wide
                top = dep_at[r, c] * deep
                if d < hwb:
                    bank = s + 0.05 + top * (d - hw) / (hwb - hw)
                else:
                    bank = s + 0.05 + top + (d - hwb) * BANK_RISE
                if g > bank:
                    f = 1.0 - u
                    f = f * f * (3.0 - 2.0 * f)
                    g = g - f * (g - bank)
            h[r, c] = g


BANK_SLOPE = 0.3          # pond banks: about 17 degrees
BANK_M = 60.0             # the farthest the bank reaches from the shore


@nb.njit(cache=True)
def carve_ponds(h, x0, y0, step, ponds, water, ident):
    """Carve ponds into a detail window. ponds rows: x, y, a, b (semi-axes), angle,
    water level, depth, kind (3 = oxbow: a crescent, its inner side bitten out toward
    the river, which lies along +v). The bed is a bowl under the level; cells under the
    level are marked in `water`, and in `ident` with the pond's row + 1. The shore wanders around the ellipse (a few angular
    harmonics with phases from the pond's position), within 1.25 of its semi-axes.
    Around the shore the ground is cut down to a bank rising BANK_SLOPE per metre from
    just above the level until it meets the ground (fading out only in the last metres of
    BANK_M, which the placement's rim tolerance keeps out of reach): the bowl alone would leave a vertical
    wall wherever the ground at the shore stands above the water (most ponds sit on
    gently sloping ground, whose high side is metres above the level)."""
    ny, nx = h.shape
    for k in range(ponds.shape[0]):
        px, py, a, b, ang, level, depth, kind = ponds[k]
        ca = np.cos(ang)
        sa = np.sin(ang)
        f1 = (px * 0.0137 + py * 0.0291) % 6.2832
        f2 = (px * 0.0213 - py * 0.0117) % 6.2832
        reach = 1.25 * a + BANK_M
        c0 = int((px - reach - x0) / step) - 1
        c1 = int((px + reach - x0) / step) + 2
        r0 = int((py - reach - y0) / step) - 1
        r1 = int((py + reach - y0) / step) + 2
        if c1 < 0 or r1 < 0 or c0 >= nx or r0 >= ny:
            continue
        for r in range(max(r0, 0), min(r1, ny)):
            wy = y0 + r * step - py
            for c in range(max(c0, 0), min(c1, nx)):
                wx = x0 + c * step - px
                u = wx * ca + wy * sa
                v = -wx * sa + wy * ca
                th = np.arctan2(v / b, u / a)
                wob = 1.0 + 0.13 * np.sin(3.0 * th + f1) + 0.08 * np.sin(5.0 * th + f2)
                rho = ((u / a) ** 2 + (v / b) ** 2) / (wob * wob)
                if rho >= 1.0:
                    # the bank: distance beyond the shore along the ray from the centre
                    dist = np.sqrt(wx * wx + wy * wy)
                    out = dist - dist / np.sqrt(rho)
                    if out < BANK_M:
                        bank = level + 0.3 + BANK_SLOPE * out
                        if h[r, c] > bank:
                            f = min((BANK_M - out) / 10.0, 1.0)
                            f = f * f * (3.0 - 2.0 * f)
                            h[r, c] -= f * (h[r, c] - bank)
                    continue
                if kind == 3.0:
                    ri = (u / (0.85 * a)) ** 2 + ((v - 0.55 * b) / (0.8 * b)) ** 2
                    if ri < 1.0:
                        # the ground the crescent curls around gets the same bank
                        inside = (1.0 - np.sqrt(ri)) * 0.8 * b
                        bank = level + 0.3 + BANK_SLOPE * inside
                        if h[r, c] > bank:
                            h[r, c] = bank
                        continue
                    rho = max(rho, 1.0 - (ri - 1.0) * 1.5)
                bed = level - depth * (1.0 - rho)
                if bed < h[r, c]:
                    h[r, c] = bed
                if h[r, c] < level:
                    water[r, c] = True
                    ident[r, c] = k + 1
