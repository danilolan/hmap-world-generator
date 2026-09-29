"""Stage 4 — relief, before erosion (WORLDGEN.md section 2).

Macro relief on the preview grid, in metres, built from the terrain character:
- land rises gently from the coast inland, with broad regional swells;
- continental drainage structure from the plates: a continent tilts up toward
  its collision zone and down toward its passive coasts (rivers born in the
  ranges cross the whole continent to the far sea, as the Amazon does from the
  Andes), and the weight of each range bends the crust beside it into a
  foreland basin, a trough parallel to the range that gathers its rivers
  (Ganges, Po): thin elastic plate flexure solved with FFTs;
- young ranges: height from the uplift, with ridged noise for crests, spurs
  and peaks (erosion, stage 5, will cut the valleys);
- plateaus: a raised, nearly flat block with an escarpment edge;
- old hill country: rounded domes and serras;
- rift valleys sink the land;
- oceanic islands (arcs, hotspot chains) rise as volcanic cones;
- sea floor: a shallow shelf off the coasts, a slope to the deep ocean,
  mid-ocean ridges where oceanic plates part, trenches where they dive.

Micro relief (detail(), evaluated at any point from world coordinates, so the
detail window and the 2 m export agree): hills a character climbs in minutes,
scaled by the character of the region, and the rock's own forms where stage 3
made the rock rugged, after how each rock weathers:
- layered rock (sandstone, limestone, basalt flows) breaks
  into cliff bands: gently dipping strata, of which the hard layers stand up
  as rock walls while the soft ones weather into slopes (differential erosion
  of layered terrain, Benes & Forsbach 2001), with woods above and below; each
  wall faces downhill, so no hollow forms behind it;
- granite rises in tors and knobs: rounded outcrops with steep sides;
- limestone sinks in dolines, the small closed hollows of karst;
- shale, and any smooth region, keeps only the soft hills.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps, raster
from ..core.memo import memo
from ..core.params import Float
from ..core.pointnoise import fbm_at, fbm_unit
from ..core.stage import Stage
from .character import ROCK_TYPES
from .land import BIG, CONT, ISLAND, ISLET
from .plates import CONVERGENT, DIVERGENT, OCEAN


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Relief(Stage):
    id = "relief"
    title = "Relief"
    description = ("Heights before erosion: ranges along collisions, plateaus, old rounded hill country, rift valleys, "
                   "lowlands, volcanic islands and the sea floor. Click the map for the detail view (micro relief).")
    params = [
        Float("range_m", "Mountain height (m)", 3200, 800, 5000, 50, "Height of the strongest ranges' peaks"),
        Float("plateau_m", "Plateau height (m)", 1300, 300, 3000, 25),
        Float("hills_m", "Hill country height (m)", 380, 50, 1200, 10, "Height of the old, rounded serras and hills"),
        Float("lowland_m", "Lowland height (m)", 120, 10, 500, 5, "How high the land rises from the coast to the interior plains"),
        Float("micro_hills_m", "Micro relief: hills (m)", 24, 0, 80, 1,
              "Small hills a character climbs in minutes (scaled by the region: bigger in hill and mountain country)"),
        Float("scarp_m", "Micro relief: rock wall height (m)", 14, 0, 40, 0.5,
              "Height of the rock walls (cliff bands) where hard layered rock is rugged"),
        Float("scarps", "Micro relief: rock walls", 0.6, 0.0, 1.0, 0.01,
              "How many of the rock layers stand up as walls in rugged layered rock"),
        Float("tor_m", "Micro relief: tors (m)", 9, 0, 30, 0.5, "Height of the rounded granite outcrops"),
        Float("doline_m", "Micro relief: sinkholes (m)", 6, 0, 20, 0.5, "Depth of the karst hollows in limestone"),
        Float("scarp_deg", "Micro relief: rock wall steepness (°)", 55, 30, 75, 1,
              "Steepest part of a rock wall's face; its top and foot round off into the slopes around it",
              advanced=True),
        Float("tilt_m", "Continental tilt (m)", 250, 0, 800, 10,
              "How much higher a continent stands near its collision zone than at its far, passive coasts: rivers "
              "born in the ranges gather into long rivers crossing the continent"),
        Float("foreland", "Foreland basins", 0.6, 0.0, 1.5, 0.01,
              "How much the weight of the ranges bends the crust beside them into troughs that gather their rivers"),
        Float("tilt_km", "Tilt length (km)", 25, 5, 60, 1,
              "Distance from the collision zone over which the continent comes down", advanced=True),
        Float("flexure_km", "Crust flexural length (km)", 5, 1, 20, 0.25,
              "Stiffness of the crust under the ranges: longer = wider, shallower foreland basins", advanced=True),
        Float("volcano_m", "Volcanic island height (m)", 900, 100, 3000, 25, "Peak height of arc and hotspot islands", advanced=True),
        Float("rift_m", "Rift valley depth (m)", 250, 0, 800, 10, advanced=True),
        Float("mountain_coast_km", "Mountains' distance to fall to the sea (km)", 3.0, 0.5, 10, 0.1,
              "Over how many km a range or plateau comes down to sea level at a coast", advanced=True),
        Float("inland_km", "Coast-to-interior rise distance (km)", 8, 1, 25, 0.5, advanced=True),
        Float("swell_m", "Regional swells (m)", 60, 0, 300, 5, "Broad rises and dips across the lowlands", advanced=True),
        Float("shelf_m", "Continental shelf depth (m)", 80, 10, 300, 5, advanced=True),
        Float("shelf_km", "Continental shelf width (km)", 2.0, 0.2, 8, 0.1, advanced=True),
        Float("floor_m", "Deepest water drawn (m)", 50, 10, 1000, 5,
              "Below this depth (sea and lakes) the bottom is a flat floor the game never draws: players cannot dive "
              "that deep, so nothing is generated there (owner's rule; a provisional value, in real metres)"),
        Float("deep_m", "Deep ocean depth (m)", 900, 200, 1000, 10,
              "Depth of the ocean shape before the floor cuts it (the height format bottoms out at -1000 m)", advanced=True),
        Float("shore_slope", "Shore slope (m per km)", 12, 2, 60, 0.5,
              "Rise of the land from the waterline (and fall of the sea floor) right at the coast", advanced=True),
        Float("shore_km", "Shore zone width (km)", 0.6, 0.2, 2.0, 0.05, advanced=True),
        Float("coast_detail_m", "Fine coastline detail (m)", 0.9, 0.0, 4.0, 0.05,
              "Small coves and points added to the coast in the detail view and the export", advanced=True),
        Float("micro_wavelength_m", "Micro hill size (m)", 650, 100, 1500, 10, "Spacing of the small hills", advanced=True),
    ]
    views = {"relief": "Relief", "height": "Height (grey)"}
    height_output = "height_m"
    has_detail = True

    def run(self, ctx, inputs, p):
        seed = ctx.stage_seed(self.id)
        n, cell = ctx.res, ctx.cell_m
        km = 1000.0 / cell
        land, coast, cls = inputs["land"], inputs["coast_dist_km"], inputs["land_class"]
        wy, wo, wp = inputs["w_young"], inputs["w_old"], inputs["w_plateau"]
        yy, xx = np.mgrid[0:n, 0:n]
        X, Y = (xx + 0.5) * cell, (yy + 0.5) * cell

        # --- land
        K, H = ctx.km, ctx.height
        base = H(p["lowland_m"]) * _smoothstep(np.maximum(coast, 0) / K(p["inland_km"])) ** 0.7
        base += H(p["swell_m"]) * np.clip(0.5 + 0.35 * fbm_unit(X, Y, K(9000), 3, seed + 1), 0, 1.2)
        ridged = fbm_at(X, Y, K(4000), 5, seed + 2, "ridged", 0.55)
        # mountains and plateaus need room to come down to the sea: a 3 km range cannot
        # stand 300 m from the water (it would be an 80° wall)
        to_coast = _smoothstep(np.maximum(coast, 0) / K(p["mountain_coast_km"])) ** 0.6
        ranges = H(p["range_m"]) * inputs["uplift"] ** 0.8 * np.clip(0.62 + 0.6 * ridged, 0.15, 1.1) * to_coast
        plateau = H(p["plateau_m"]) * _smoothstep((wp - 0.25) / 0.5) * (0.9 + 0.08 * fbm_unit(X, Y, K(2500), 3, seed + 3)) * to_coast
        domes = np.clip(0.5 + 0.4 * fbm_unit(X, Y, K(3000), 4, seed + 4, 0.45), 0, 1) ** 1.6
        hills = H(p["hills_m"]) * wo * domes
        rift = -H(p["rift_m"]) * inputs["rift"]
        oceanic = (cls == ISLAND) | (cls == ISLET)
        volcanic = np.where(oceanic, H(p["volcano_m"]) * _smoothstep(np.maximum(coast, 0) / K(1.2)) ** 1.3, 0)
        relief = np.maximum(ranges, np.maximum(plateau, hills)) + 0.3 * np.minimum(ranges, hills)
        base = base + self._drainage_structure(ctx, p, inputs, land, cls, coast, ranges + plateau, base)
        h_land = base + relief + rift + volcanic

        # --- sea floor
        plate, kind, btype = inputs["plate"], inputs["plate_kind"], inputs["boundary_type"]
        lo, hi = raster.boundary_pairs(plate)
        klo, khi = kind[np.maximum(lo, 0)], kind[np.maximum(hi, 0)]
        dsea = np.maximum(-coast, 0)
        depth = p["shelf_m"] * _smoothstep(dsea / p["shelf_km"]) + \
            (p["deep_m"] - p["shelf_m"]) * _smoothstep((dsea - p["shelf_km"]) / 6.0)
        depth *= np.clip(1.0 + 0.12 * fbm_unit(X, Y, 6000, 3, seed + 5), 0.7, 1.3)
        ridge = (lo >= 0) & (btype == DIVERGENT) & (klo == OCEAN) & (khi == OCEAN)
        if ridge.any():
            depth -= 0.45 * p["deep_m"] * np.exp(-((ndimage.distance_transform_edt(~ridge) / km) / 2.5) ** 2)
        trench = (lo >= 0) & (btype == CONVERGENT) & ((klo == OCEAN) | (khi == OCEAN))
        if trench.any():
            depth += 0.3 * p["deep_m"] * np.exp(-((ndimage.distance_transform_edt(~trench) / km) / 1.0) ** 2)
        h_sea = -np.clip(depth, 2.0, 1000.0)

        h = np.where(land, h_land, h_sea)
        # near the shore the height follows a smooth ramp of the (slightly blurred)
        # signed distance to the coast, so the zero contour is a smooth curve at any
        # sampling resolution instead of the preview grid's staircase
        sd = ndimage.gaussian_filter(coast, 1.0)
        ramp = p["shore_slope"] * sd
        wgt = 1.0 - _smoothstep(np.abs(sd) / K(p["shore_km"]))
        h = h * (1 - wgt) + ramp * wgt
        h = np.where(land, np.maximum(h, 0.2), np.clip(h, -p["floor_m"], -0.2)).astype(np.float32)
        return {"height_m": h, "relief_params": dict(p)}

    @staticmethod
    def _drainage_structure(ctx, p, inputs, land, cls, coast, load, base):
        """Height added to the lowlands by the continents' tilt and the foreland basins.
        Tilt: continental land rises toward the collision zone (the young ranges) and
        comes down toward the coasts far from it; near every coast it still starts from
        the coastal ramp, so it only adds height inland. Foreland basins: the ranges'
        weight deflects a thin elastic plate, w(k) = load(k) * (rho_c / rho_m) /
        (1 + (alpha k)^4 / 4); the deflection beside the ranges (not under them, whose
        heights are set by the range knobs) sinks the lowlands into troughs, never below
        a few metres above the sea."""
        K, H = ctx.km, ctx.height
        km = 1000.0 / ctx.cell_m
        n = ctx.res
        cont = land & np.isin(cls, (CONT, BIG))
        add = np.zeros((n, n))
        young = inputs["uplift"] > 0.2
        if not (cont & young).any():
            return add
        # tilt: up toward the collision zone, reaching zero tilt_km away from it
        d = ndimage.distance_transform_edt(~young) / km
        # the tilt spans the continent: it is measured on the map, not shrunk with the world scale
        tilt = H(p["tilt_m"]) * (1.0 - _smoothstep(d / p["tilt_km"]))
        inland = _smoothstep(np.maximum(coast, 0) / K(p["inland_km"]))
        add += np.where(cont, tilt * inland, 0.0)
        # foreland basins: flexure of a thin elastic plate under the ranges' load
        ky = np.fft.fftfreq(n, d=ctx.cell_m) * 2 * np.pi
        kx = np.fft.rfftfreq(n, d=ctx.cell_m) * 2 * np.pi
        k = np.hypot(ky[:, None], kx[None, :])
        alpha = K(p["flexure_km"]) * 1000.0
        w = np.fft.irfft2(np.fft.rfft2(load) / (1.0 + (alpha * k) ** 4 / 4.0), s=(n, n)) * (2.7 / 3.3)
        beside = 1.0 - _smoothstep(load / max(H(150.0), 1e-3))
        trough = p["foreland"] * np.maximum(w, 0.0) * beside
        lowland = base + add
        add -= np.where(cont, np.minimum(trough, np.maximum(lowland - H(8.0), 0.0) * 0.9), 0.0)
        # never below a few metres above the sea: the lowlands stay land
        return np.maximum(add, -np.maximum(base - H(8.0), 0.0))

    # ------------------------------------------------------------------ micro relief
    def detail(self, ctx, data, p, x_m, y_m):
        return micro_relief(ctx, data, p, x_m, y_m, data["height_m"])

    def render(self, view, ctx, data):
        h = data["height_m"]
        if view == "height":
            return colormaps.gray(h, -1000, 4000)
        return colormaps.terrain(h, ctx.cell_m)

    def stats(self, ctx, data):
        h, land = data["height_m"], data["land"]
        hl = h[land]
        return {"highest (m)": int(hl.max()) if hl.size else 0, "land median (m)": int(np.median(hl)) if hl.size else 0,
                "land above 1000 m %": round(float((hl > 1000).mean() * 100), 1) if hl.size else 0,
                "deepest (m)": int(h.min()), **climb_stats(h, ctx.cell_m)}


def climb_stats(h, cell_m):
    """How long a walker takes up the highest peak, by Naismith's rule (5 km/h plus one hour
    per 600 m of ascent). The foot is the nearest point below 15 % of the peak's height; the
    distance is the straight line from there, so real paths take somewhat longer."""
    iy, ix = np.unravel_index(int(np.argmax(h)), h.shape)
    peak = float(h[iy, ix])
    if peak <= 50:
        return {}
    yy, xx = np.indices(h.shape)
    dist = np.hypot(yy - iy, xx - ix) * cell_m / 1000.0

    def reach(level):
        below = h < level
        return float(dist[below].min()) if below.any() else 0.0
    foot, d_foot = 0.15 * peak, reach(0.15 * peak)
    half = foot + 0.5 * (peak - foot)
    d_half = d_foot - reach(half)
    minutes = lambda d, up: round((d / 5.0 + up / 600.0) * 60)
    return {"climb to highest peak (min)": minutes(d_foot, peak - foot),
            "to its half height (min)": minutes(d_half, half - foot)}


def _cell_range(grid, coords):
    """Lowest and highest of the four grid samples around each point (the ones bilinear
    interpolation would blend), so a smoother sampling can be kept within them."""
    n0, n1 = grid.shape
    fy, fx = np.floor(coords[0]).astype(np.int64), np.floor(coords[1]).astype(np.int64)
    y0, y1 = np.clip(fy, 0, n0 - 1), np.clip(fy + 1, 0, n0 - 1)
    x0, x1 = np.clip(fx, 0, n1 - 1), np.clip(fx + 1, 0, n1 - 1)
    a, b, c, d = grid[y0, x0], grid[y0, x1], grid[y1, x0], grid[y1, x1]
    return np.minimum(np.minimum(a, b), np.minimum(c, d)), np.maximum(np.maximum(a, b), np.maximum(c, d))


def _grid_slope(grid, cell):
    gy, gx = np.gradient(np.asarray(grid, np.float32), cell)
    return np.hypot(gx, gy)


def keep_side(h, delta):
    """h + delta without carrying a point across sea level: the part of `delta` that
    pushes toward 0 is compressed smoothly (tanh) so it never takes more than 90 % of
    |h|. Small changes pass unchanged. A hard clamp at the coast instead would leave
    every dip of low coastal land as one dead-flat sheet at the clamp height."""
    lim = 0.9 * np.abs(h)
    toward = np.where(h > 0, -delta, delta)                    # > 0: toward sea level
    squeezed = lim * np.tanh(np.maximum(toward, 0.0) / np.maximum(lim, 1e-6))
    return h + np.where(toward > 0, np.where(h > 0, -squeezed, squeezed), delta)


def micro_relief(ctx, data, p, x_m, y_m, height_grid):
    """Heights at world coordinates: `height_grid` (the macro relief on the preview
    grid, before or after erosion) sampled there, plus the micro relief of stage 4
    (`p` are stage 4's parameters). Shared by the relief and erosion stages and,
    later, the 2 m export."""
    seed = ctx.stage_seed(Relief.id)
    cell = ctx.cell_m
    coords = [y_m / cell - 0.5, x_m / cell - 0.5]
    # cubic sampling, kept within the range of the surrounding cells: a plain cubic spline
    # overshoots at steep steps (a 200 m cliff next to the sea rings into phantom land
    # strips offshore and moats inland), the same reason the game's surface is monotone
    h = ndimage.map_coordinates(height_grid, coords, order=3, mode="nearest")
    # the spline may overshoot its four surrounding samples a little (clamping it exactly
    # would flatten every local low or high into a square plateau), but never by much and
    # never across sea level where the plain bilinear sampling does not cross it
    lo, hi = _cell_range(height_grid, coords)
    tol = 0.1 * (hi - lo) + 0.5
    h = np.clip(h, lo - tol, hi + tol)
    hb = ndimage.map_coordinates(height_grid, coords, order=1, mode="nearest")
    h = np.where((h > 0) != (hb > 0), hb, h)
    w = {k: ndimage.map_coordinates(data["w_" + k].astype(np.float32), coords, order=1, mode="nearest")
         for k in ("young", "old", "plateau", "plain")}
    # fine coastline irregularity (coves, points) the preview grid cannot hold
    # (only right at the coastline: on a wide, shallow shelf it would raise strips of
    # phantom land far offshore)
    near_coast = 1 - _smoothstep(np.abs(ndimage.map_coordinates(data["coast_dist_km"].astype(np.float32), coords, order=1,
                                                                mode="nearest")) * 1000.0 / 80.0)
    # It moves the waterline sideways by at most ~6 m: the height change is capped by the
    # local slope, so on a nearly flat sea bottom (a sandy shore) it cannot raise lines and
    # rings of land that enclose pools
    grid_slope = memo(data, ("slope", id(height_grid)), lambda: _grid_slope(height_grid, cell))
    grad = ndimage.map_coordinates(grid_slope, coords, order=1, mode="nearest")
    amp = np.minimum(p["coast_detail_m"], 6.0 * grad + 0.02)
    h = h + amp * fbm_unit(x_m, y_m, 350, 2, seed + 13) * (1 - _smoothstep(np.abs(h) / 12.0)) * near_coast
    land = h > 0.0
    at = lambda a: ndimage.map_coordinates(np.asarray(a, np.float32), coords, order=1, mode="nearest")
    rugged = at(data["rugged"]) if "rugged" in data else np.zeros(np.shape(x_m))
    rock = {r: at(data["rock_w"][i]) for i, r in enumerate(ROCK_TYPES)} if "rock_w" in data else None
    # small hills, scaled by region, craggier where the rock is rugged
    scale = (0.8 * w["plain"] + 1.5 * w["old"] + 1.0 * w["plateau"] + 2.2 * w["young"]) * (0.8 + 0.5 * rugged)
    micro = p["micro_hills_m"] * scale * 0.5 * fbm_unit(x_m, y_m, p["micro_wavelength_m"], 4, seed + 10, 0.45)
    micro += p["micro_hills_m"] * 0.25 * rugged * (fbm_at(x_m, y_m, 220, 3, seed + 14, "ridged", 0.5) - 0.35)
    # the macro slope, point by point from the grid, so any set of points (a window, an
    # export tile, a pond's rim) agrees
    slope = grad
    if rock is not None:
        # cliff bands: gently dipping strata (a regional dip, a warp so a wall's line
        # wanders ~60 m across the slope instead of following the contours, and ridged
        # re-entrants where gullies cut back into it, scaled by the slope so the wander is
        # the same in metres on any ground); the hard
        # layers stand up as walls, the soft ones weather into slopes. q counts layers up
        # the slope: a wall of scarp_m every ~2.5 walls' height of rise, so the ground
        # between walls still falls downhill and nothing is trapped behind a wall
        layered = rock["sandstone"] + rock["limestone"] + rock["basalt"]          # granite has no layers
        dip = 0.02 * fbm_unit(x_m, y_m, 9000, 2, seed + 15)
        z = h + dip * x_m + 0.02 * fbm_unit(x_m, y_m, 9000, 2, seed + 16) * y_m \
            + 10.0 * fbm_unit(x_m, y_m, 500, 3, seed + 17) \
            + np.maximum(slope, 0.03) * (60.0 * fbm_unit(x_m, y_m, 220, 3, seed + 23)
                                         + 20.0 * fbm_at(x_m, y_m, 80, 2, seed + 24, "ridged", 0.5))
        spacing = 2.5 * max(p["scarp_m"], 1e-3)
        # layers of uneven thickness (a monotone wobble of z: two incommensurate waves, thin
        # layers 0.6x and thick ones 2.5x the mean), so walls do not stack at one regular
        # spacing up a long slope; dq is how much faster the layers pass than on average
        zs = z / spacing
        w1, w2 = 0.61 * zs * 2 * np.pi + 1.3, 0.23 * zs * 2 * np.pi + 4.1
        q = zs + 0.12 * np.sin(w1) + 0.1 * np.sin(w2)
        dq = 1.0 + 0.12 * 0.61 * 2 * np.pi * np.cos(w1) + 0.1 * 0.23 * 2 * np.pi * np.cos(w2)
        layer = np.floor(q)
        frac = q - layer
        hard = (np.abs(np.sin(layer * 12.9898 + seed % 1000) * 43758.5453) % 1.0) < p["scarps"] * 0.7
        # a wall runs for a stretch, then breaks (gullies, buttresses, slopes of scree):
        # segments of one to a few hundred metres, of varying height
        stretch = _smoothstep((fbm_unit(x_m, y_m, 260, 3, seed + 18) + 0.1) / 0.35)
        # the riser's share of a layer, from the face steepness wanted: the wall climbs
        # scarp_m over riser * spacing / slope metres of ground (the warp about doubles how
        # fast z climbs, hence 2 slope), and a smoothstep face is 1.5 times steeper at its
        # middle than on average; toward a segment's end the face
        # also widens, so the wall dies out as a slope instead of ending in a sheer pillar
        face = np.tan(np.radians(p["scarp_deg"])) / 1.5
        riser = np.clip(p["scarp_m"] * 2.0 * dq * np.maximum(slope, 0.02) / (face * spacing) / (0.3 + 0.7 * stretch),
                        0.04, 0.6)
        wall = p["scarp_m"] * (_smoothstep((frac - (1 - riser)) / riser) - frac) * hard
        # moderate slopes: on steep mountainsides the walls would crowd into thin hatching
        on_slope = _smoothstep((slope - 0.04) / 0.06) * (1.0 - _smoothstep((slope - 0.4) / 0.25))
        band = rugged * _smoothstep((layered - 0.2) / 0.4) * on_slope
        micro += wall * band * stretch * (0.6 + 0.5 * _smoothstep((fbm_unit(x_m, y_m, 700, 2, seed + 22) + 1.0) / 2.0))
        # tors: rounded granite (and some basalt) outcrops with steep sides
        k = fbm_unit(x_m, y_m, 90, 2, seed + 19)
        tors = _smoothstep((k - 1.3) / 0.5) * (0.7 + 0.3 * fbm_unit(x_m, y_m, 30, 2, seed + 20))
        micro += p["tor_m"] * tors * rugged * (rock["granite"] + 0.4 * rock["basalt"]) * (1.0 - _smoothstep((slope - 0.5) / 0.3))
        # dolines: the closed hollows of karst limestone (dry: the water sinks underground)
        dl = fbm_unit(x_m, y_m, 160, 2, seed + 21)
        micro -= p["doline_m"] * _smoothstep((dl - 1.2) / 0.4) * rugged * rock["limestone"] * (1.0 - _smoothstep(slope / 0.25))
    # keep the coastline where stage 2 put it
    fade = np.where(land, _smoothstep(h / 6.0), 0.15)
    out = keep_side(h, micro * fade)
    return np.where(land, np.maximum(out, 0.01), np.clip(out, -p["floor_m"], -0.01)).astype(np.float32)
