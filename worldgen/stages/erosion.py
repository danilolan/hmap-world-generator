"""Stage 5 — erosion, rivers and lakes (WORLDGEN.md section 2).

A landscape-evolution loop on the preview grid. The land starts low and is
uplifted toward the stage 4 relief while it erodes (uplift against erosion,
Cordonnier et al. 2016): that competition is what grows branching valley
networks and sharp ridges. Heights are matched back to stage 4 region by region
at the end. Each iteration:
1. route the water (priority-flood, depressions filled for routing);
2. drainage area with multiple flow directions (no straight grooves);
3. rivers incise their beds (implicit stream power, Braun & Willett 2013);
4. the eroded material travels downstream and settles where the river can no
   longer carry it: flat valley floors, fans at mountain fronts, deltas;
5. soil creeps down the hillslopes (nonlinear diffusion, Roering 1999):
   convex crests, planar slopes;
6. slopes steeper than the talus angle collapse (landslides, scree).
Every process is scaled by the terrain character of stage 3: young ranges get
deep incision and little rounding (sharp crests, V valleys); old hill country
gets strong rounding (soft serras); plateaus resist (their escarpments survive).

After the loop: large, deep depressions become lakes (a flat water level);
every other depression is breached, so small bowls become valleys. Rivers are
traced from the drainage area as smoothed centrelines with a width that grows
downstream; the detail window (and later the 2 m export) carves them as
channels and adds gullies on steep slopes.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps, hydro
from ..core import lines as lines_mod
from ..core.memo import memo
from ..core.params import Float, Int
from ..core.pointnoise import fbm_at
from ..core.stage import Stage
from .relief import climb_stats, keep_side, micro_relief

# per character: incision, rounding, talus angle
CHAR_K = {"young": 1.0, "old": 0.3, "plateau": 0.35, "plain": 0.4}
CHAR_KD = {"young": 0.02, "old": 1.0, "plateau": 0.1, "plain": 0.4}


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def routing_roughness(ctx, p, land):
    """A little roughness for routing only: on smooth ramps water would otherwise run in
    straight parallel lines instead of gathering into river systems. Shared with the
    hydrology stage so its streams follow the same valleys."""
    n, dx = ctx.res, ctx.cell_m
    yy, xx = np.mgrid[0:n, 0:n]
    rough = ctx.height(p["route_roughness_m"]) * fbm_at((xx + 0.5) * dx, (yy + 0.5) * dx, 1500, 4,
                                                        ctx.stage_seed(Erosion.id) + 5)
    return np.where(land, rough, 0.0)


def lake_water(data, h, lake_id, lake_level, height, coords):
    """Which lake (id, 0 = none) each point of a detail window lies under: the lake's
    level reaches up to two grid cells onto its shore (not down an outlet lower than it),
    so the shoreline is the contour where the ground meets the level instead of the
    grid's blocky lake cells."""
    def reach():
        near, (iy, ix) = ndimage.distance_transform_edt(lake_id == 0, return_indices=True)
        lvl, ids = lake_level[iy, ix], lake_id[iy, ix]
        ok = ((lake_id > 0) | ((near <= 2) & (height >= lvl - 0.5))) & (ids > 0)
        return np.where(ok, lvl, -np.inf), np.where(ok, ids, 0).astype(np.float32)
    level, ids = memo(data, ("lake_reach", id(lake_id)), reach)
    under = h < ndimage.map_coordinates(level, coords, order=0, mode="nearest")
    return np.where(under, np.rint(ndimage.map_coordinates(ids, coords, order=0, mode="nearest")), 0).astype(np.int32)


def ground(ctx, data, p, x_m, y_m):
    """Heights at world coordinates after erosion, before any water is carved: the eroded
    macro relief, stage 4's micro relief and gullies on steep slopes (`p` are stage 5's
    parameters). Shared by the erosion and hydrology detail and, later, the 2 m export."""
    h = micro_relief(ctx, data, data["relief_params"], x_m, y_m, data["height_eroded"])
    coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
    at = lambda a: ndimage.map_coordinates(a.astype(np.float32), coords, order=1, mode="nearest")
    slope, wy, wo, wp = at(data["slope"]), at(data["w_young"]), at(data["w_old"]), at(data["w_plateau"])
    # gullies: a ridged network cut into steep slopes
    g = fbm_at(x_m, y_m, 160, 4, ctx.stage_seed(Erosion.id) + 1, "ridged", 0.5)
    amp = p["gully_m"] * _smoothstep((slope - 0.2) / 0.5) * (wy + 0.6 * wo + 0.5 * wp)
    return np.where(h > 0, np.maximum(keep_side(h, amp * 2.0 * (g - 0.35)), 0.01), h).astype(np.float32)


class Erosion(Stage):
    id = "erosion"
    title = "Erosion, rivers and lakes"
    description = ("Rivers carve valleys, sediment fills valley floors and builds fans and deltas, hillslopes round off, "
                   "steep slopes slide; big trapped basins become lakes. Intensity follows the terrain character.")
    params = [
        Int("iterations", "Erosion time", 120, 10, 400, 5, "Iterations of the simulation: longer = deeper valleys, more mature landscape"),
        Float("incision", "River incision", 1.0, 0.0, 3.0, 0.01, "How hard rivers cut into the rock"),
        Float("rounding", "Hillslope rounding", 1.0, 0.0, 3.0, 0.01, "How much soil creep softens crests and slopes"),
        Float("deposition", "Sediment deposition", 1.0, 0.0, 3.0, 0.01,
              "How readily rivers drop their load: flat valley floors, fans and deltas"),
        Float("talus_deg", "Landslide angle, mountains (°)", 46, 25, 60, 0.5,
              "Steepest slope young mountains keep before sliding (other terrains use gentler angles)"),
        Float("river_km2", "River threshold (km²)", 3.0, 0.3, 30, 0.1,
              "Drainage area a stream needs to count as a river: lower = more, smaller rivers"),
        Float("lake_km2", "Lakes: minimum area (km²)", 0.8, 0.05, 10, 0.05, "Smaller trapped basins are drained instead"),
        Float("lake_depth_m", "Lakes: minimum depth (m)", 6, 1, 100, 0.5),
        Float("meander", "River meandering", 1.0, 0.0, 3.0, 0.01, "How much rivers wind across flat land"),
        Float("gully_m", "Gullies on steep slopes (m)", 8, 0, 30, 0.5,
              "Depth of the small gullies the detail view adds to steep slopes"),
        Float("uplift", "Uplift during erosion", 2.0, 0.0, 5.0, 0.05,
              "How much the land rises while rivers cut it: this grows branching valleys and sharp ridges", advanced=True),
        Float("start_fraction", "Starting height (fraction of relief)", 0.3, 0.0, 1.0, 0.01, advanced=True),
        Float("match_window_km", "Height matching window (km)", 3.0, 0.5, 10, 0.1,
              "Region size over which heights are matched back to the relief stage", advanced=True),
        Float("m", "Area exponent (m)", 0.45, 0.3, 0.7, 0.01, "Stream power: how much more big rivers cut than small ones", advanced=True),
        Float("k_base", "Incision rate", 30.0, 1.0, 80.0, 0.5, advanced=True),
        Float("kd_base", "Creep rate (m² per iteration)", 700, 0, 3000, 10, advanced=True),
        Float("critical_deg", "Creep critical slope (°)", 38, 20, 50, 0.5,
              "Slope where soil creep runs away (hillslopes become planar near it)", advanced=True),
        Float("capacity", "River carrying capacity", 40, 1, 400, 1, advanced=True),
        Float("fan_slope", "Steepest fan / valley floor slope", 0.03, 0.002, 0.15, 0.001, advanced=True),
        Float("mfd_exponent", "Flow spreading exponent", 1.1, 0.5, 4.0, 0.05,
              "Lower = water spreads over more neighbours (smoother), higher = narrower flow", advanced=True),
        Int("route_every", "Reroute water every N iterations", 2, 1, 5, advanced=True),
        Float("route_roughness_m", "Routing roughness (m)", 15.0, 0.0, 60.0, 0.5,
              "Small bumps that make water gather into river systems on smooth ramps", advanced=True),
    ]
    views = {"relief": "Relief", "water": "Rivers and lakes", "change": "Erosion / deposition", "drainage": "Drainage area"}
    units = {'relief': 'm', 'water': 'm', 'drainage': 'log10 km2'}
    height_output = "height_eroded"
    has_detail = True

    def run(self, ctx, inputs, p):
        n, dx = ctx.res, ctx.cell_m
        h0 = inputs["height_m"].astype(np.float64)
        land = inputs["land"]
        # uplift against erosion (Cordonnier et al. 2016): the land starts low and rises
        # toward the stage 4 relief while the rivers cut it, which is what grows
        # branching valleys and sharp ridges; heights are matched back afterwards
        h = np.where(land, p["start_fraction"] * h0, h0)
        U = np.where(land, np.maximum(h0, 0) * p["uplift"] / max(p["iterations"], 1), 0.0)
        w = {k: inputs["w_" + k] for k in CHAR_K}
        K = p["k_base"] * p["incision"] * sum(w[k] * CHAR_K[k] for k in CHAR_K)
        kd = p["kd_base"] / ctx.scale ** 2 * p["rounding"] * sum(w[k] * CHAR_KD[k] for k in CHAR_KD)
        talus_deg = p["talus_deg"] * w["young"] + 48 * w["plateau"] + 27 * w["old"] + 22 * w["plain"]
        # arc and hotspot islands are young volcanic cones: hard lava, little soil
        volcanic_island = np.isin(inputs["land_class"], (3, 4))
        talus = np.tan(np.radians(np.where(volcanic_island, 42.0, talus_deg)))
        K = np.where(volcanic_island, 0.5 * K, K)
        kd = np.where(volcanic_island, 0.1 * kd, kd)
        # a little roughness for routing only: on smooth ramps water would otherwise
        # run in straight parallel lines instead of gathering into river systems
        rough = routing_roughness(ctx, p, land)
        sc = np.tan(np.radians(p["critical_deg"]))
        steps = int(np.ceil(kd.max() * 5.0 / (dx * dx) / 0.2)) if kd.max() > 0 else 0
        capacity = p["capacity"] / max(p["deposition"], 1e-3)
        deposited = np.zeros(n * n)
        eroded_total = np.zeros(n * n)
        route = None
        for it in range(p["iterations"]):
            if route is None or it % p["route_every"] == 0:
                hf, rec, dist, order = hydro.route(h + rough, land)
                hf = hf - rough.ravel()
                area = hydro.mfd_area(hf + rough.ravel(), order, land, dx * dx, p["mfd_exponent"])
                route = True
            h += U
            e = hydro.incise(h, hf, rec, dist, order, area, K, land, dx, p["m"])
            eroded_total += e
            hydro.transport(h, rec, dist, order, area, e, land, dx, p["m"], capacity, p["fan_slope"], deposited, 2.0)
            if steps:
                hydro.creep(h, kd, land, dx, sc, steps)
            if it % 4 == 0:
                h = hydro.landslide(h, talus, land, dx, 0.5)
        # match heights back to stage 4 region by region (crest windows): the shape comes
        # from the simulation, the height hierarchy from the relief stage
        r = max(1, int(round(ctx.km(p["match_window_km"]) * 1000.0 / dx)))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        hl, h0l = np.where(land, h, 0.0), np.where(land, h0, 0.0)
        cur = ndimage.gaussian_filter(ndimage.grey_dilation(hl, footprint=disk), r / 2)
        tgt = ndimage.gaussian_filter(ndimage.grey_dilation(h0l, footprint=disk), r / 2)
        ratio = ndimage.gaussian_filter(np.clip(tgt / np.maximum(cur, 1.0), 0.3, 6.0), r / 2)
        h = np.where(land, h * ratio, h)
        h = np.where(land, np.maximum(h, 0.2), np.minimum(h, -0.2))

        # lakes: big, deep trapped basins keep water; the rest are breached
        hf, rec, dist, order = hydro.route(h, land)
        hf2 = hf.reshape(n, n)
        depth = np.where(land, hf2 - h, 0)
        lab, cnt = ndimage.label(depth > 0.3)
        keep = np.zeros((n, n), bool)
        lake_id = np.zeros((n, n), np.int32)
        lakes = []
        min_cells = p["lake_km2"] / ctx.scale ** 2 * 1e6 / (dx * dx)
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab[sl] == k
            if m.sum() < min_cells or depth[sl][m].max() < ctx.height(p["lake_depth_m"]):
                continue
            keep[sl] |= m
            lid = len(lakes) + 1
            lake_id[sl][m] = lid
            ys, xs = np.nonzero(m)
            lakes.append(dict(id=lid, level_m=float(hf2[sl][m].max()), area_km2=float(m.sum() * dx * dx / 1e6),
                              max_depth_m=float(depth[sl][m].max()),
                              fetch_km=float(max(np.ptp(xs), np.ptp(ys)) + 1) * dx / 1000.0))
        tmp = np.where(keep, hf2, h)
        hydro.breach(tmp, land, keep)
        h = np.where(keep, h, tmp)
        # lakes deeper than the floor the game draws end in the same flat floor
        floor = inputs["relief_params"]["floor_m"]
        for L in lakes:
            m = lake_id == L["id"]
            h[m] = np.maximum(h[m], L["level_m"] - floor)
            L["max_depth_m"] = min(L["max_depth_m"], floor)
        level = np.zeros((n, n), np.float32)
        for L in lakes:
            level[lake_id == L["id"]] = L["level_m"]

        # rivers: single-flow drainage on the final surface
        hf, rec, dist, order = hydro.route(np.where(keep, level, h) + rough, land)
        area = hydro.sfd_area(rec, order, dx * dx)
        gy, gx = np.gradient(h, dx)
        slope = np.hypot(gx, gy)
        segs, lines = self._rivers(area * ctx.scale ** 2, rec, land, lake_id, n, dx, p, slope, ctx.stage_seed(self.id), ctx.scale)
        return {"height_eroded": h.astype(np.float32), "drainage_km2": (area.reshape(n, n) / 1e6).astype(np.float32),
                "lake_id": lake_id, "lake_level": level, "lakes": lakes, "river_segments": segs, "river_count": len(lines),
                "erosion_change": (h - h0).astype(np.float32), "slope": slope.astype(np.float32),
                "erosion_params": dict(p)}

    @staticmethod
    def _rivers(area, rec, land, lake_id, n, dx, p, slope, seed, scale=1.0):
        """`area` in the world's own units (km² of the uncompressed world); widths are
        walker scale and not shrunk by the world scale."""
        thr = p["river_km2"] * 1e6
        lf = land.ravel()
        is_river = (area > thr) & lf & (lake_id.ravel() == 0)
        idx = np.nonzero(is_river)[0]
        has_donor = np.zeros(n * n, bool)
        tgt = rec[idx]
        has_donor[tgt[tgt != idx]] = True
        visited = np.zeros(n * n, bool)
        segs, lines = [], []
        for s in idx[~has_donor[idx]]:
            pts, c = [], s
            while True:
                i, j = divmod(int(c), n)
                a_km2 = area[c] / 1e6
                pts.append(((j + 0.5) * dx, (i + 0.5) * dx, float(np.clip(3.0 * np.sqrt(a_km2), 1.5, 120.0))))
                r = rec[c]
                if visited[c] or r == c or not lf[r] or lake_id.ravel()[r]:
                    if r != c and not visited[c]:
                        ri, rj = divmod(int(r), n)
                        pts.append(((rj + 0.5) * dx, (ri + 0.5) * dx, pts[-1][2]))
                    visited[c] = True
                    break
                visited[c] = True
                c = r
            if len(pts) >= 2:
                sm = lines_mod.meander(lines_mod.chaikin(pts, 3), slope, dx, p["meander"], seed + len(lines))
                lines.append(sm)
                for a, b in zip(sm[:-1], sm[1:]):
                    wdt = 0.5 * (a[2] + b[2])
                    segs.append((a[0], a[1], b[0], b[1], 0.5 * wdt, float(np.clip(0.6 + 0.05 * wdt, 0.6, 4.0))))
        return np.array(segs, np.float64).reshape(-1, 6), lines

    # ------------------------------------------------------------------ detail
    def detail(self, ctx, data, p, x_m, y_m):
        h = ground(ctx, data, p, x_m, y_m).astype(np.float64)
        coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
        # river channels
        bed = np.zeros(h.shape, np.bool_)
        segs = data["river_segments"]
        if len(segs):
            hydro_carve(h, bed, x_m, y_m, segs)
        # water: lakes below their level, river beds
        water = (lake_water(data, h, data["lake_id"], data["lake_level"], data["height_eroded"], coords) > 0) | bed
        return h.astype(np.float32), water

    # ------------------------------------------------------------------ views
    def legend(self, view, ctx, data):
        if view == "change":
            return [{"color": [200, 60, 40], "label": "eroded (up to 200 m)"}, {"color": [60, 170, 80], "label": "deposited"}]
        return None

    def legend_labels(self, view, ctx, data):
        if view != "change":
            return None
        c = data["erosion_change"]
        return np.where(data["land"] & (np.abs(c) > 2.0), np.where(c < 0, 0, 1), -1).astype(np.int16)

    def render(self, view, ctx, data):
        h = data["height_eroded"]
        if view == "change":
            c = data["erosion_change"]
            t = np.clip(np.abs(c) / 200.0, 0, 1)[..., None]
            col = np.where((c < 0)[..., None], np.array([200, 60, 40]), np.array([60, 170, 80]))
            img = (np.full(h.shape + (3,), 235.0) * (1 - t) + col * t)
            img[~data["land"]] = (40, 75, 125)
            return img.astype(np.uint8)
        if view == "drainage":
            return colormaps.gray(np.log10(np.maximum(data["drainage_km2"], 1e-3)), -2, 2.5)
        img = colormaps.terrain(h, ctx.cell_m)
        if view == "water":
            a = data["drainage_km2"]
            thr = self.coerce({})["river_km2"]
            for lim, grow in ((thr, 0), (thr * 8, 1), (thr * 40, 2)):
                m = (a > lim) & data["land"]
                if grow:
                    m = ndimage.binary_dilation(m, iterations=grow) & data["land"]
                img[m] = (40, 105, 215)
            img[data["lake_id"] > 0] = (60, 130, 220)
        return img

    def stats(self, ctx, data):
        c, land = data["erosion_change"], data["land"]
        return {"rivers": int(data["river_count"]), "lakes": len(data["lakes"]),
                "deepest cut (m)": int(-c[land].min()) if land.any() else 0,
                "thickest deposit (m)": int(c[land].max()) if land.any() else 0,
                "highest (m)": int(data["height_eroded"].max()), **climb_stats(data["height_eroded"], ctx.cell_m)}


def hydro_carve(h, bed, x_m, y_m, segs):
    """Carve river channels into a detail window (rounded cross-section; where
    segments overlap the deepest cut wins)."""
    from ..core.raster_carve import carve_segments
    x0, y0 = float(x_m[0, 0]), float(y_m[0, 0])
    step = float(x_m[0, 1] - x_m[0, 0])
    cut = carve_segments(h.shape[0], h.shape[1], x0, y0, step, segs, bed)
    h -= cut
