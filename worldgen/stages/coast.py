"""Stage 11 — coast (WORLDGEN.md section 2).

The shoreline's character, from coastal geomorphology: waves erode where they
strike hard (exposed headlands), sediment settles where they are weak (bays),
rivers feed the beaches, and the land behind decides between a beach and a
cliff.
- wave exposure: the open-water fetch in sixteen directions from every
  coastal point, weighted toward the prevailing wind over the sea (stage 6),
  plus a share of swell from every direction;
- shape: bays (land around the water) shelter, headlands expose;
- backshore: how high the land stands a little inland (cliffs need height);
- sediment: sandy soils and rivers deliver sand; young, rocky country delivers
  pebbles; rivers' mouths deliver mud where sheltered.
Coast types: sand beach, shingle beach (pebbles), rocky shore, cliff, salt
marsh and mudflat, estuary. Rivers wide enough end in estuaries: drowned
valleys that widen seaward in a funnel (the width grows exponentially toward
the mouth, as in real estuaries), with the water at sea level.
In the detail window (and the export) the ground takes the coast's shape:
beaches become gentle ramps of sand or pebbles, cliffs rise steeply from the
water with a rock platform at their foot, salt marshes stay just above the
tide.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps
from ..core.memo import memo
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.stage import Stage
from .erosion import ground
from .hydrology import water_detail

COASTS = [
    ("none", (0, 0, 0)),
    ("sand beach", (235, 215, 150)),
    ("shingle beach", (175, 165, 150)),
    ("rocky shore", (120, 115, 110)),
    ("cliff", (200, 70, 60)),
    ("salt marsh / mudflat", (120, 140, 95)),
    ("estuary", (70, 150, 200)),
]
NONE, SAND, SHINGLE, ROCKY, CLIFF, MARSH, ESTUARY = range(len(COASTS))
DIRECTIONS = 16


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Coast(Stage):
    id = "coast"
    title = "Coast"
    description = ("Beaches in bays, pebble beaches by rocky country, rocky shores and cliffs on exposed headlands, salt "
                   "marshes where sheltered, and funnel-shaped estuaries at the mouths of the larger rivers.")
    params = [
        Float("cliffs", "Cliffs", 1.0, 0.0, 2.0, 0.01, "How readily high, exposed coasts become cliffs"),
        Float("beach_m", "Beach width (m)", 35, 5, 150, 1, "Typical width of a beach from the waterline to the land"),
        Float("estuary_min_m", "Estuaries for rivers from (m wide)", 5.0, 1.0, 30.0, 0.5,
              "Rivers at least this wide at the sea end in an estuary"),
        Float("estuary", "Estuary size", 1.0, 0.2, 3.0, 0.01,
              "How wide the estuaries open toward the sea (their mouths reach tens of river widths)"),
        Float("marsh", "Salt marshes", 1.0, 0.0, 2.0, 0.01, "How much of the sheltered low coast is salt marsh"),
        Float("fetch_km", "Longest fetch counted (km)", 20, 5, 60, 1,
              "Open water beyond this adds no more to the waves", advanced=True),
    ]
    views = {"coast": "Coast types", "exposure": "Wave exposure", "bathymetry": "Sea depth"}
    units = {'exposure': 'sheltered - exposed', 'bathymetry': 'depth (m)'}
    height_output = "height_eroded"
    has_detail = True

    # ------------------------------------------------------------------ run
    def run(self, ctx, inputs, p):
        n, dx = ctx.res, ctx.cell_m
        seed = ctx.stage_seed(self.id)
        land = inputs["land"]
        h = inputs["height_eroded"].astype(np.float64)
        sea = ~land & (inputs["water_lake_id"] == 0)
        shore_sea = sea & ndimage.binary_dilation(land, iterations=1)
        shore_land = land & ndimage.binary_dilation(sea, iterations=1)

        exposure_sea = self._exposure(ctx, inputs, p, sea, shore_sea)
        # carry each shore sea cell's exposure to the land within a few cells
        _, (iy, ix) = ndimage.distance_transform_edt(~shore_sea, return_indices=True)
        exposure = exposure_sea[iy, ix]
        # bays shelter, headlands expose: how much land surrounds each point (1 km)
        around = ndimage.gaussian_filter(land.astype(np.float64), ctx.km(1000.0) / dx)
        exposure = exposure * np.clip(1.6 - 1.2 * around, 0.4, 1.6)
        # backshore height: the land a little inland
        back = ndimage.grey_dilation(np.where(land, h, 0.0), size=5)
        back_h = np.where(land, back, 0.0)
        coarse = inputs.get("sand", np.zeros((n, n)))
        young = inputs["w_young"] + 0.6 * inputs["w_plateau"]
        yy, xx = np.mgrid[0:n, 0:n]
        X, Y = (xx + 0.5) * dx, (yy + 0.5) * dx
        noise = fbm_unit(X, Y, 1500, 3, seed + 1)

        ctype = np.zeros((n, n), np.int8)
        zone = land & (np.maximum(inputs["coast_dist_km"], 0) < 0.3)
        # exposure ranked against the rest of this world's shore (0 = most sheltered, 1 = most exposed)
        shore_e = np.sort(exposure[shore_land]) if shore_land.any() else np.array([0.0, 1.0])
        rank = np.interp(exposure, shore_e, np.linspace(0.0, 1.0, shore_e.size))
        r = np.clip(rank + 0.12 * noise, 0, 1)
        hard = np.clip(inputs["w_young"] + 0.6 * inputs["w_plateau"] + 0.4 * inputs["w_old"], 0, 1)
        # cliffs where high land meets strong waves, in any rock (chalk cliffs are soft)
        cliff = _smoothstep((back_h - ctx.height(15.0)) / ctx.height(60.0)) + 0.8 * r + 0.2 * noise > 1.45 / max(p["cliffs"], 1e-3)
        rocky = (0.7 * r + 0.5 * hard + 0.2 * noise > 0.62) & ~cliff
        shingle = (0.5 * r + 0.6 * hard + 0.2 * noise > 0.55)
        marsh = (r < 0.22 * p["marsh"]) & (back_h < ctx.height(12.0))
        # where the land rises steeply from the water (a mountain or a hill running into
        # the sea) no beach forms: rocky shore
        near_back = ndimage.grey_dilation(np.where(land, h, 0.0), size=3)
        steep = near_back > ctx.height(30.0) + ctx.height(15.0) * noise
        ctype[zone] = SAND
        ctype[zone & shingle] = SHINGLE
        # sandy country feeds sand
        ctype[zone & (ctype == SHINGLE) & (coarse > 0.6)] = SAND
        ctype[zone & (rocky | steep)] = ROCKY
        # pocket beaches: sheltered coves under steep coasts keep a small beach of pebbles
        ctype[zone & steep & ~rocky & (r < 0.35)] = SHINGLE
        ctype[zone & cliff] = CLIFF
        ctype[zone & marsh] = MARSH

        estuaries, mouths = self._estuaries(ctx, inputs, p, land)
        for x, y, w in mouths:
            i, j = int(y / dx), int(x / dx)
            r = max(1, int(round(4 * w / dx)))
            sl = (slice(max(i - r, 0), i + r + 1), slice(max(j - r, 0), j + r + 1))
            ctype[sl][zone[sl]] = np.where(exposure[sl][zone[sl]] < 0.8, MARSH, ctype[sl][zone[sl]])
        # ponds levelled on the ground before the coast took its shape would pit the new
        # beaches and cliff terraces: keep only those away from shaped coasts
        pd = inputs["ponds"]
        if len(pd):
            i = np.clip((pd[:, 1] / dx).astype(int), 0, n - 1)
            j = np.clip((pd[:, 0] / dx).astype(int), 0, n - 1)
            shaped = ndimage.binary_dilation(np.isin(ctype, (SAND, SHINGLE, CLIFF, MARSH)), iterations=3)
            pd = pd[~shaped[i, j]]
        return {"coast_type": ctype, "wave_exposure": exposure.astype(np.float32), "backshore_m": back_h.astype(np.float32),
                "estuary_segments": estuaries, "estuary_mouths": mouths, "coast_ponds": np.ascontiguousarray(pd, np.float64),
                "coast_params": dict(p)}

    @staticmethod
    def _exposure(ctx, data, p, sea, shore_sea):
        """Open-water fetch from each shore sea cell in sixteen directions (km of the
        uncompressed world), weighted toward the wind over the sea; normalised to ~1."""
        n, dx = ctx.res, ctx.cell_m
        ys, xs = np.nonzero(shore_sea)
        if not ys.size:
            return np.zeros((n, n))
        step = max(1.0, n / 256.0)
        max_steps = int(p["fetch_km"] * 1000.0 / ctx.scale / (dx * step)) + 1
        wy, wx = data["wind_dy"][ys, xs], data["wind_dx"][ys, xs]
        total = np.zeros(ys.size)
        wsum = 0.0
        for k in range(DIRECTIONS):
            a = 2 * np.pi * k / DIRECTIONS
            dy, dx_ = np.sin(a), np.cos(a)
            fetch = np.zeros(ys.size)
            open_ = np.ones(ys.size, bool)
            for s in range(1, max_steps + 1):
                yi = np.clip((ys + dy * s * step).astype(int), 0, n - 1)
                xi = np.clip((xs + dx_ * s * step).astype(int), 0, n - 1)
                open_ &= sea[yi, xi]
                fetch += open_
                if not open_.any():
                    break
            fetch_km = fetch * step * dx * ctx.scale / 1000.0
            # waves arrive from where the wind blows from: the direction opposite to its travel
            toward = np.clip(-(dy * wy + dx_ * wx), 0, 1)
            w = 0.3 + toward
            total += w * np.sqrt(fetch_km)
            wsum += 1.0
        e = np.zeros((n, n))
        e[ys, xs] = total / wsum / np.sqrt(p["fetch_km"]) * 1.4
        return e

    @staticmethod
    def _estuaries(ctx, data, p, land):
        """Funnel-shaped estuaries at the sea mouths of rivers at least estuary_min_m wide:
        segments in the carving format (surface at sea level, width growing
        exponentially toward the mouth)."""
        dx = ctx.cell_m
        n = land.shape[0]
        segs, mouths = [], []
        for line in data["stream_lines"]:
            line = np.asarray(line, np.float64)
            x, y, w = line[-1, 0], line[-1, 1], line[-1, 2]
            i, j = int(np.clip(y / dx, 0, n - 1)), int(np.clip(x / dx, 0, n - 1))
            if land[i, j] or w < p["estuary_min_m"]:
                continue
            w_mouth = min(w * 25.0 * p["estuary"], 2500.0)
            length = 5.0 * w_mouth
            converge = length / 3.0
            dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(line[::-1, 0]), np.diff(line[::-1, 1])))])[::-1]
            keep = dist <= length
            if keep.sum() < 2:
                continue
            pts = line[keep]
            dd = dist[keep]
            width = w + (w_mouth - w) * np.exp(-dd / converge)
            depth = 1.0 + 0.004 * width
            a, b = pts[:-1], pts[1:]
            wa = 0.5 * (width[:-1] + width[1:])
            da = 0.5 * (depth[:-1] + depth[1:])
            reach = 0.5 * wa * 2.0 + 60.0
            segs.append(np.stack([a[:, 0], a[:, 1], b[:, 0], b[:, 1], 0.5 * wa, da, np.zeros(len(a)), np.zeros(len(a)),
                                  reach, np.zeros(len(a))], -1))
            mouths.append((x, y, w_mouth))
        segs = np.ascontiguousarray(np.concatenate(segs), np.float64) if segs else np.zeros((0, 10))
        return segs, np.array(mouths).reshape(-1, 3)

    # ------------------------------------------------------------------ detail
    def detail(self, ctx, data, p, x_m, y_m):
        h, water, overlays, shore = self.shore(ctx, data, p, x_m, y_m)
        land = (h > 0) & ~water
        colors = {"sand": (230, 212, 150), "shingle": (170, 162, 150), "cliff": (125, 118, 112),
                  "rocky": (120, 115, 110), "marsh": (110, 135, 95)}
        mats = [(shore[k] & land, c) for k, c in colors.items()]
        return h, water, overlays + [(m, c) for m, c in mats if m.any()]

    TYPES = (SAND, SHINGLE, ROCKY, CLIFF, MARSH)

    @staticmethod
    def _type_weights(ctx, data):
        """Per coast type, a smoothed grid mask of the nearest shore's type carried a little
        out to sea (computed once per data dict)."""
        grid_type = data["coast_type"]
        has = grid_type > 0
        if not has.any():
            return None
        dist_c, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
        ext = np.where(dist_c * ctx.cell_m < 400.0, grid_type[iy, ix], 0)
        return {t: ndimage.gaussian_filter((ext == t).astype(np.float32), 1.5) for t in Coast.TYPES}

    def shore(self, ctx, data, p, x_m, y_m, ids=None):
        """The final ground of a regular window: stage 5's ground with each coast's profile
        imposed across the shore, tidied at the waterline, and stage 7's water (plus the
        estuaries and the ponds that survive the shaping) carved in. Returns (heights,
        water mask, overlays, shore materials: dict of masks "sand", "shingle" (beach
        pebbles), "cliff" and "rocky" (bare rock), "marsh"). Shared by the detail window
        and the 2 m export; `ids` is passed on to water_detail."""
        h = ground(ctx, data, data["erosion_params"], x_m, y_m).astype(np.float64)
        coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
        at = lambda a, o=1: ndimage.map_coordinates(np.asarray(a, np.float32), coords, order=o, mode="nearest")
        seed = ctx.stage_seed(self.id)
        n1 = fbm_unit(x_m, y_m, 80, 3, seed + 5)
        step = float(x_m[0, 1] - x_m[0, 0])
        land0 = h > 0
        grid_w = memo(data, "coast_type_weights", lambda: self._type_weights(ctx, data))
        if grid_w is None:
            h, water, overlays = water_detail(ctx, data, x_m, y_m, h, ids=ids)
            none = np.zeros(h.shape, bool)
            return h, water, overlays, {k: none for k in ("sand", "shingle", "cliff", "rocky", "marsh")}
        d_m = self._shore_distance(ctx, data, x_m, y_m, step)
        # the coast type of the nearest shore as smooth weights per type (so neighbouring
        # coast types blend instead of meeting in grid steps); each type gets its own noise,
        # so the borders between types wander along the coast instead of following the grid
        wt = {t: np.clip(at(grid_w[t]) * (1.0 + 0.5 * fbm_unit(x_m, y_m, 220, 3, seed + 30 + t)), 0, 1)
              for t in self.TYPES}
        near = sum(wt.values()) > 0.05
        ctype = np.full(h.shape, NONE, np.int8)
        best = np.zeros(h.shape)
        for t, w in wt.items():
            win = w > best
            ctype[win & (w > 0.05)] = t
            best = np.maximum(best, w)
        back = at(data["backshore_m"])
        W = p["beach_m"] * (0.7 + 0.5 * at(data["wave_exposure"])) * (0.8 + 0.3 * n1)
        # Each coast imposes its own profile across the shore, as a function of the distance
        # to the coastline, so the waterline is exactly where the profile puts it (the fine
        # coastline noise of the ground would otherwise leave moats and slivers), then blends
        # back into the ground inland and offshore.
        floor = data["relief_params"]["floor_m"]
        # beaches: a gentle ramp from a shallow sandy bottom up to a low berm, W metres wide,
        # only where the ground is low and gentle (sand does not climb a hillside); where it
        # rises steeply the shore stays the ground's own slope, rocky
        gy_, gx_ = np.gradient(h, step)
        ground_slope = ndimage.gaussian_filter(np.hypot(gx_, gy_), 3.0)
        low = (h < 3.0 + 0.1 * np.maximum(d_m, 0.0)) & (ground_slope < 0.25 + 0.05 * n1)
        beach = near & (wt[SAND] + wt[SHINGLE] > 0.02) & (d_m > -120.0) & (d_m < W * 2.0) & (low | (d_m < 0))
        steep_shore = near & np.isin(ctype, (SAND, SHINGLE)) & (d_m >= 0) & (d_m < 30.0) & ~low
        slope = np.where(ctype == SHINGLE, 1 / 8.0, 1 / 25.0)
        ramp = np.where(d_m >= 0, 0.15 + slope * d_m, 0.15 + 0.06 * d_m)
        keep_ground = np.where(d_m >= 0, _smoothstep((d_m - W) / W), _smoothstep((-d_m - 60.0) / 60.0))
        shaped = ramp * (1 - keep_ground) + h * keep_ground
        # inland the ground may not dip below the berm (no pools behind the beach), offshore
        # not rise above the sandy bottom
        # inland the ground may not dip below the berm (no pools behind the beach); offshore
        # the bottom keeps its depth, only nothing may stand above the water
        shaped = np.where((d_m >= 0) & land0, np.maximum(shaped, np.minimum(ramp, 0.15 + slope * W)), np.minimum(h, -0.1 + 0.02 * np.minimum(d_m, 0.0)))
        wb = np.clip(wt[SAND] + wt[SHINGLE], 0, 1)
        h = np.where(beach, h + (shaped - h) * _smoothstep(wb / 0.5), h)
        # cliffs: the sea has cut away the foot of the slope. The face rises from a rock
        # platform in the water to the height the ground already has where the face ends,
        # a few tens of metres inland (sampled along the coast's normal); beyond it the
        # ground is untouched, so no terrace or shelf is added
        cliff = near & (ctype == CLIFF) & (d_m > -300.0 + 40.0 * n1) & (d_m < 120.0)
        if cliff.any():
            ny, nx = np.gradient(ndimage.gaussian_filter(d_m, 4.0))
            norm = np.maximum(np.hypot(nx, ny), 1e-9)
            reach = 25.0 + 15.0 * (n1 + 1.0)                           # where the face ends
            ahead = np.clip(reach - d_m, 0.0, reach)                    # how far inland the top is taken
            top = ground(ctx, data, data["erosion_params"], x_m + nx / norm * ahead, y_m + ny / norm * ahead)
            # smoothed along the coast: in concave corners the inland samples converge on one
            # spot and would raise wedges
            top = ndimage.gaussian_filter(np.clip(np.maximum(top, h), 0.0, ctx.height(200.0)), 30.0 / step)
            face = reach
            rise = 0.3 + top * _smoothstep((d_m - 3.0) / np.maximum(face - 3.0, 1.0))
            profile = np.where(d_m >= 0, rise, np.maximum(-0.5 + 0.12 * d_m, -floor))
            # only where there is height to make a cliff
            wc = _smoothstep(wt[CLIFF] / 0.5) * _smoothstep((top - ctx.height(8.0)) / ctx.height(15.0))
            # raise only ground that is land already (near the window's edge the grid distance
            # may call a bit of sea land)
            target = np.where(d_m >= 0, np.where((d_m < face) & land0, np.maximum(profile, h), h),
                              h + (np.minimum(profile, h) - h) * (1 - _smoothstep((-d_m - 150.0) / 150.0)))
            h = np.where(cliff, h + (target - h) * wc, h)
        else:
            face = np.zeros(h.shape)
        # salt marsh: flat, just above the tide
        marsh = near & (ctype == MARSH) & (d_m > -20.0) & (d_m < 180.0 + 80.0 * n1)
        h = np.where(marsh & (h > 0), np.minimum(h, 0.3 + 0.8 * _smoothstep(d_m / 250.0) + 0.2 * n1), h)
        h = self._tidy_waterline(h, step)
        h, water, overlays = water_detail(ctx, data, x_m, y_m, h, data["estuary_segments"], data["coast_ponds"], ids)
        # shore materials, on land and on the sea bottom along the shore
        shore = {"sand": beach & (ctype == SAND), "shingle": beach & (ctype == SHINGLE),
                 "cliff": cliff & (d_m < face + 20.0), "rocky": (near & (ctype == ROCKY) & (d_m < 40.0)) | steep_shore,
                 "marsh": marsh}
        return h, water, overlays, shore

    @staticmethod
    def _tidy_waterline(h, step, pool_m2=3000.0, sliver_m2=600.0):
        """The shore's small changes of height right at sea level (fine coastline noise,
        beach, marsh and cliff profiles) can cut off little pools of sea on land and leave
        slivers of land in the water. Pools of sea smaller than pool_m2 not reaching the
        window's edge become land just above the tide; slivers of land smaller than
        sliver_m2 become sea. (Lakes, ponds and rivers are water above sea level or carved
        later, so they are untouched.)"""
        out = h.copy()
        for is_sea, limit, fix in ((True, pool_m2, 0.15), (False, sliver_m2, -0.15)):
            mask = (out <= 0) if is_sea else (out > 0)
            lab, n = ndimage.label(mask)
            if n < 2:
                continue
            sizes = ndimage.sum(np.ones_like(out), lab, range(1, n + 1)) * step * step
            border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            small = np.nonzero(sizes < limit)[0] + 1
            small = small[~np.isin(small, border)]
            if small.size:
                m = np.isin(lab, small)
                out[m] = np.maximum(out[m], fix) if is_sea else np.minimum(out[m], fix)
        return out

    @staticmethod
    def _shore_distance(ctx, data, x_m, y_m, step, margin=400.0):
        """Signed distance (m) from each point to the coastline of the ground itself (> 0
        land, < 0 sea), so coves and inlets narrower than the preview grid are respected.
        The ground is sampled coarsely over the window plus a margin, so points whose
        nearest coast lies outside the window still find it (the export does the same with
        its tiles' margins); distances beyond the margin are only lower bounds, which the
        shore profiles never reach."""
        coarse = max(step, 4.0)
        # on a lattice fixed in the world, so overlapping windows (the export's blocks)
        # measure the same distances
        x0 = np.floor((float(x_m[0, 0]) - margin) / coarse) * coarse
        y0 = np.floor((float(y_m[0, 0]) - margin) / coarse) * coarse
        nx = int(np.ceil((float(x_m[0, -1]) - float(x_m[0, 0]) + 2 * margin) / coarse)) + 2
        ny = int(np.ceil((float(y_m[-1, 0]) - float(y_m[0, 0]) + 2 * margin) / coarse)) + 2
        gx, gy = np.meshgrid(x0 + np.arange(nx) * coarse, y0 + np.arange(ny) * coarse)
        land = ground(ctx, data, data["erosion_params"], gx, gy) > 0
        d = np.where(land, ndimage.distance_transform_edt(land) - 0.5, -ndimage.distance_transform_edt(~land) + 0.5) * coarse
        return ndimage.map_coordinates(d, [(y_m - y0) / coarse, (x_m - x0) / coarse], order=1, mode="nearest")

    # ------------------------------------------------------------------ views
    def legend(self, view, ctx, data):
        if view == "coast":
            return [{"color": list(c), "label": n} for n, c in COASTS[1:]]
        if view == "detail":
            return [{"color": [230, 212, 150], "label": "sand beach"}, {"color": [170, 162, 150], "label": "pebble beach"},
                    {"color": [125, 118, 112], "label": "cliff rock"}, {"color": [120, 115, 110], "label": "rocky shore"},
                    {"color": [110, 135, 95], "label": "salt marsh"}, {"color": [55, 115, 200], "label": "water"}]
        return None

    def render(self, view, ctx, data):
        land = data["land"]
        h = data["height_eroded"]
        if view == "exposure":
            img = colormaps.heat(data["wave_exposure"], 0.0, 1.6)
            img[land & (data["coast_type"] == 0)] = (90, 110, 80)
            return img.astype(np.uint8)
        if view == "bathymetry":
            img = colormaps.heat(-np.minimum(h, 0), 0, 1000)[..., ::-1].copy()
            img[land] = (90, 110, 80)
            return img.astype(np.uint8)
        img = colormaps.terrain(h, ctx.cell_m).astype(np.float64)
        ct = data["coast_type"]
        wide = ndimage.grey_dilation(ct, size=3) * (ct == 0) + ct
        for t in range(1, len(COASTS)):
            m = (wide == t) & land
            img[m] = COASTS[t][1]
        for x, y, w in data["estuary_mouths"]:
            i, j = int(y / ctx.cell_m), int(x / ctx.cell_m)
            r = max(1, int(round(w / ctx.cell_m)))
            img[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1][~land[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]] = COASTS[ESTUARY][1]
        return img.astype(np.uint8)

    def stats(self, ctx, data):
        ct = data["coast_type"]
        shore = data["land"] & ndimage.binary_dilation(~data["land"] & (data["water_lake_id"] == 0))
        types = ct[shore]
        types = types[types > 0]
        if not types.size:
            return {}
        share = np.bincount(types, minlength=len(COASTS)) / types.size * 100
        mouths = data["estuary_mouths"]
        cliff_h = np.minimum(data["backshore_m"][shore & (ct == CLIFF)], ctx.height(200.0))
        return {"coast": ", ".join(f"{COASTS[i][0]} {share[i]:.0f}%" for i in range(1, len(COASTS)) if share[i] >= 0.5),
                "estuaries": len(mouths), "widest estuary mouth (m)": int(mouths[:, 2].max()) if len(mouths) else 0,
                "highest cliffs (m)": int(cliff_h.max()) if cliff_h.size else 0}
