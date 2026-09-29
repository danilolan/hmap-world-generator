"""Stage 8 — soil and fertility (WORLDGEN.md section 2).

Soil from the factors that form it (Jenny's clorpt, as used in digital soil
mapping: scorpan, McBratney et al. 2003) — climate, organisms, relief, parent
material and time — every one of them already computed by earlier stages:
- parent material: a mosaic of rock types (granite and gneiss in the young
  ranges, basalt on the plateaus and volcanic islands, sandstone, shale and
  limestone in the old country and plains) under the loose deposits the other
  stages left: river alluvium sorted by the water's energy (sand and gravel by
  the channels, silt and clay on the floodplains), colluvium at the foot of
  slopes, lake clays, coastal dune sand, and loess, the silt the wind blows out
  of dry country and drops downwind (the fertile belts of Europe and China);
- depth: soil production against removal (Heimsath et al. 1997): thin on
  steep slopes and convex crests, deep in hollows and footslopes, thickened by
  deposits, deeper where the climate weathers rock faster;
- texture (sand, silt, clay) from the parent material, plus clay from
  weathering; organic matter from the vegetation the climate supports (most in
  grasslands, peat where the ground is waterlogged and cool); nutrients from
  base-rich parents, lost to leaching where much water drains through;
  drainage from the wetness index and the height above the nearest channel;
  salt where water evaporates in closed basins.
From these, a soil group (a simplified World Reference Base) and a fertility
for farming. Poor and rich soils interleave along every catena (crest, slope,
footslope, valley floor), so fertile ground is never far away; the stats check
the owner's rule that it lies within a 30-minute walk of any infertile spot.
The detail window shows the 2 m tile materials that the export will write
(rock, gravel, sand, clay, dirt; bare, grass or dry grass).
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.stage import Stage
from .land import ISLAND, ISLET
from .hydrology import Hydrology

# parent rocks (in stage 3's ROCK_TYPES order): sand, silt, clay fractions of the soil they
# weather into, and base richness
ROCKS = {
    "granite": ((0.60, 0.25, 0.15), 0.35),
    "sandstone": ((0.72, 0.18, 0.10), 0.25),
    "shale": ((0.20, 0.42, 0.38), 0.55),
    "limestone": ((0.25, 0.40, 0.35), 0.90),
    "basalt": ((0.30, 0.42, 0.28), 0.85),
}
DEPOSITS = {
    "alluvium_coarse": ((0.75, 0.18, 0.07), 0.70),
    "alluvium_fine": ((0.15, 0.50, 0.35), 0.85),
    "colluvium": ((0.40, 0.35, 0.25), 0.65),
    "lake": ((0.10, 0.40, 0.50), 0.70),
    "dune": ((0.92, 0.05, 0.03), 0.20),
    "loess": ((0.12, 0.72, 0.16), 0.90),
}
SOILS = [  # id order is the view's legend
    ("none", (40, 75, 125)),
    ("Leptosol (thin, rocky)", (150, 145, 140)),
    ("Regosol (young, weak)", (200, 185, 150)),
    ("Cambisol (brown earth)", (160, 130, 80)),
    ("Luvisol (clay-rich, fertile)", (175, 110, 60)),
    ("Podzol (acid, sandy)", (205, 205, 190)),
    ("Chernozem (black earth)", (60, 50, 40)),
    ("Kastanozem (dry grassland)", (140, 105, 70)),
    ("Fluvisol (alluvial)", (95, 140, 110)),
    ("Gleysol (waterlogged)", (95, 120, 150)),
    ("Histosol (peat)", (70, 70, 55)),
    ("Andosol (volcanic)", (105, 75, 90)),
    ("Arenosol (sand)", (230, 215, 160)),
    ("Solonchak (salty)", (235, 235, 235)),
]
SOIL_ID = {name.split(" ")[0]: i for i, (name, _) in enumerate(SOILS)}
FERTILE = 0.5            # fertility from which farming is efficient
ROCK, GRAVEL, SAND, CLAY, DIRT = 4, 3, 1, 2, 0     # TileGround values (TERRAIN.md section 2.1)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Soil(Stage):
    id = "soil"
    title = "Soil and fertility"
    description = ("Soil from its forming factors: rock and deposits, depth along the slopes, texture, organic matter, "
                   "nutrients, drainage and salt; soil groups and farming fertility; the 2 m tile materials in the "
                   "detail window.")
    params = [
        Float("soil_depth_m", "Soil depth on gentle ground (m)", 1.2, 0.3, 3.0, 0.05,
              "Typical depth where the ground is gentle; crests and steep slopes get less, hollows and valley floors more"),
        Float("weathering", "Weathering", 1.0, 0.2, 2.0, 0.01,
              "How fast the climate turns rock into soil and clay (faster where warm and wet)"),
        Float("alluvium", "River deposits", 1.0, 0.0, 2.0, 0.01, "How much of the valley floors is rich alluvium"),
        Float("loess", "Wind-blown silt (loess)", 0.6, 0.0, 2.0, 0.01,
              "Silt the wind lifts from dry country and drops downwind: deep, fertile soils"),
        Float("organic", "Organic matter", 1.0, 0.0, 2.0, 0.01,
              "How rich the topsoil gets under grassland and forest (dark, fertile soils where high)"),
        Float("leaching", "Leaching", 1.0, 0.0, 2.0, 0.01,
              "How much the water draining through washes nutrients out (acid, poor soils where wet and sandy)"),
        Float("rock_depth_m", "Bare rock below (m)", 0.12, 0.02, 0.5, 0.01,
              "Where the soil is thinner than this the tile is exposed rock", advanced=True),
        Float("fertile_min", "Fertile from", FERTILE, 0.2, 0.9, 0.01,
              "Fertility from which farming counts as efficient (for the walking rule)", advanced=True),
        Float("walk_min", "Rule: fertile land within (minutes)", 30, 5, 90, 1,
              "Walking at 5 km/h, fertile land must be this close to any infertile spot", advanced=True),
    ]
    views = {"fertility": "Fertility", "soil": "Soil groups", "depth": "Soil depth", "texture": "Texture",
             "parent": "Parent material", "reach": "Distance to fertile land"}
    units = {'depth': 'm'}
    height_output = "height_eroded"
    has_detail = True

    # ------------------------------------------------------------------ run
    def run(self, ctx, inputs, p):
        n, dx = ctx.res, ctx.cell_m
        seed = ctx.stage_seed(self.id)
        land = inputs["land"]
        h = inputs["height_eroded"].astype(np.float64)
        yy, xx = np.mgrid[0:n, 0:n]
        X, Y = (xx + 0.5) * dx, (yy + 0.5) * dx
        K = ctx.km
        w = {k: inputs["w_" + k] for k in ("young", "plateau", "old", "plain")}
        volcanic = np.isin(inputs["land_class"], (ISLAND, ISLET))
        temp, rain, aridity = inputs["temp_mean"], inputs["rain_mm"], inputs["aridity"]
        runoff = inputs["runoff_mm"]
        slope = inputs["slope"].astype(np.float64)
        change = inputs["erosion_change"]
        hand = inputs["hand_m"]
        twi = inputs["wetness_index"]
        wet = inputs["wetland"]
        salt_flat = inputs["salt_flat"]
        lake = inputs["water_lake_id"] > 0
        channel = inputs["channel_class"] > 0

        # --- parent rock: stage 3's mosaic of rock types
        rock_w = inputs["rock_w"].astype(np.float64)

        # --- loose deposits
        depo = np.clip(change, 0.0, None)
        near_ch, (ci, cj) = ndimage.distance_transform_edt(~channel, return_indices=True)
        near_ch = near_ch * dx
        # a floodplain is as wide as its river is big (tens of widths), low above it and flat
        plain_m = 25.0 * inputs["discharge_m3s"][ci, cj] ** 0.5 * 7.0 + 40.0
        floodplain = (1.0 - _smoothstep(np.maximum(hand, 0) / ctx.height(3.0))) * (1.0 - _smoothstep(slope / 0.06)) \
            * (1.0 - _smoothstep((near_ch - plain_m) / (0.5 * plain_m)))
        fan = _smoothstep((depo - ctx.height(4.0)) / ctx.height(10.0)) * (1.0 - _smoothstep(slope / 0.12))
        alluvial = np.clip(p["alluvium"] * np.maximum(floodplain, fan), 0, 1)
        coarse = alluvial * (1.0 - _smoothstep(near_ch / 60.0)) * (0.4 + 0.6 * _smoothstep(slope / 0.06))
        fine = alluvial - coarse
        hs = ndimage.gaussian_filter(h, 1.0)
        curv = ndimage.laplace(hs) / (dx * dx)                          # > 0 concave (hollows), < 0 convex
        cnorm = curv / max(float(np.std(curv[land])) if land.any() else 1.0, 1e-9)
        colluvium = (1.0 - alluvial) * _smoothstep(cnorm / 1.5) * _smoothstep(slope / 0.05) * (1.0 - _smoothstep((slope - 0.35) / 0.2))
        lake_bed = (salt_flat | ndimage.binary_dilation(lake, iterations=1) & ~lake).astype(np.float64)
        coast = np.maximum(inputs["coast_dist_km"], 0.0)
        dunes = (1.0 - _smoothstep(coast / K(0.8))) * _smoothstep((inputs["wind_exposure"] - 0.9) / 0.4) \
            * (1.0 - _smoothstep(slope / 0.05)) * _smoothstep((0.9 - aridity) / 0.6 + 0.3)
        loess = self._loess(ctx, inputs, p, land, aridity, salt_flat, alluvial)
        dep_w = {"alluvium_coarse": coarse, "alluvium_fine": fine, "colluvium": colluvium, "lake": lake_bed,
                 "dune": dunes, "loess": loess}
        total_dep = np.clip(sum(dep_w.values()), 0, 1)
        scale_dep = np.where(sum(dep_w.values()) > 1, total_dep / np.maximum(sum(dep_w.values()), 1e-9), 1.0)
        dep_w = {k: v * scale_dep for k, v in dep_w.items()}
        bed_share = 1.0 - total_dep

        # --- texture and base richness from the mix of parents
        frac = np.zeros((3, n, n))
        base = np.zeros((n, n))
        for i, r in enumerate(ROCKS):
            f, b = ROCKS[r]
            frac += bed_share * rock_w[i] * np.array(f)[:, None, None]
            base += bed_share * rock_w[i] * b
        for k, v in dep_w.items():
            f, b = DEPOSITS[k]
            frac += v * np.array(f)[:, None, None]
            base += v * b
        # weathering: warm, wet and old ground turns minerals into clay
        wx = p["weathering"] * np.clip(rain / 1000.0, 0.2, 2.0) * np.exp(0.06 * (temp - 8.0))
        age = 0.3 * w["young"] + 0.7 * w["plateau"] + 1.0 * w["old"] + 0.8 * w["plain"]
        clay_gain = 0.12 * np.clip(wx * age, 0, 1.5) * bed_share
        frac[2] += clay_gain
        frac[0] -= clay_gain * 0.7
        frac[1] -= clay_gain * 0.3
        frac = np.clip(frac, 0.02, None)
        frac /= frac.sum(0)
        sand, silt, clay = frac

        # --- depth: production against removal (Heimsath et al. 1997)
        d0 = p["soil_depth_m"] * np.clip(0.6 + 0.4 * wx, 0.3, 1.6)
        char = 0.35 * w["young"] + 0.6 * w["plateau"] + 1.3 * w["old"] + 1.1 * w["plain"]
        char = np.where(volcanic, 0.8, char)
        depth = d0 * char * np.exp(-slope / 0.35) * np.exp(0.35 * np.clip(cnorm, -3, 3))
        depth += np.clip(depo, 0, ctx.height(20.0)) * 0.15 + 1.5 * (alluvial + loess + colluvium * 0.5)
        depth *= np.clip(0.5 + 0.5 * _smoothstep(aridity / 0.6), 0.5, 1.0)
        depth = np.where(land, np.clip(depth, 0.0, 6.0), 0.0)

        # --- water, organic matter, nutrients, salt
        waterlog = np.clip(np.maximum(wet, _smoothstep((twi - 11.0) / 3.0) * (1.0 - _smoothstep(slope / 0.05))), 0, 1)
        waterlog = np.maximum(waterlog, lake_bed * 0.5)
        # grassland (steppe) climate: dense roots, dark topsoil; it needs a continental
        # climate (cold winters, warm dry summers), not a mild maritime coast
        amp = inputs["temp_summer"] - inputs["temp_winter"]
        lo_amp, hi_amp = (np.percentile(amp[land], [20, 85]) if land.any() else (0.0, 1.0))
        continental = _smoothstep((amp - lo_amp) / max(hi_amp - lo_amp, 1e-6))
        grass = np.exp(-((aridity - 0.6) / 0.25) ** 2) * (0.35 + 0.65 * continental)
        forest = _smoothstep((aridity - 0.6) / 0.6)
        om = p["organic"] * (0.25 + 0.55 * grass + 0.25 * forest) * np.clip(1.0 - 0.02 * (temp - 8.0) ** 2 / 4.0, 0.4, 1.0)
        om = np.clip(om * _smoothstep(depth / 0.3) + 0.6 * waterlog * _smoothstep((10.0 - temp) / 6.0), 0, 1.5)
        leach = p["leaching"] * _smoothstep((runoff - 250.0) / 700.0) * (0.5 + 0.8 * sand)
        nutrients = np.clip(base * (1.0 - 0.6 * leach) + 0.25 * om, 0, 1.2)
        # salt builds up where water evaporates instead of draining: closed dry basins and
        # their margins, not beside running water
        closed = ndimage.binary_dilation(salt_flat | (inputs["water_lake_id"] > 0) & (aridity < 0.5), iterations=2)
        salinity = np.clip(salt_flat * 1.0 + closed * _smoothstep((0.45 - aridity) / 0.25)
                           * (1.0 - _smoothstep(slope / 0.04)) * _smoothstep((near_ch - 150.0) / 150.0), 0, 1)
        # water held for plants: loams and silts hold most, over the rooting depth
        awc = (0.08 * sand + 0.22 * silt + 0.16 * clay) * np.minimum(depth, 1.5) / 0.25
        water = (0.35 + 0.65 * _smoothstep((aridity - 0.15) / 0.5)) * (0.6 + 0.4 * np.clip(awc, 0, 1))
        heavy = _smoothstep((clay - 0.45) / 0.15)                        # sticky clays work badly

        # --- fertility for farming
        fert = (_smoothstep(depth / 0.6) ** 0.7 * np.clip(nutrients, 0, 1) ** 0.6 * np.clip(water, 0, 1) ** 0.5
                * (1.0 - 0.75 * waterlog) * (1.0 - salinity) * (0.75 + 0.25 * np.clip(om, 0, 1)) * (1.0 - 0.3 * heavy)
                * np.clip((temp + 2.0) / 8.0, 0.2, 1.0))
        fert = np.where(land & ~lake, np.clip(fert * 1.25, 0, 1), 0.0)

        soil = self._groups(land & ~lake, depth, sand, clay, om, leach, waterlog, salinity, alluvial, volcanic, aridity,
                            temp, w, slope, wet, p)
        parent = np.argmax(np.stack([bed_share * rock_w[i] for i in range(len(ROCKS))] + list(dep_w.values())), 0)
        return {"soil_depth_m": depth.astype(np.float32), "sand": sand.astype(np.float32), "silt": silt.astype(np.float32),
                "clay": clay.astype(np.float32), "organic": om.astype(np.float32), "nutrients": nutrients.astype(np.float32),
                "waterlog": waterlog.astype(np.float32), "salinity": salinity.astype(np.float32),
                "fertility": fert.astype(np.float32), "soil_group": soil, "parent": parent.astype(np.int8),
                "alluvial": alluvial.astype(np.float32), "soil_params": dict(p)}

    @staticmethod
    def _loess(ctx, data, p, land, aridity, salt_flat, alluvial):
        """Silt lifted by the wind from dry, bare country (and dry floodplains) and dropped
        downwind with a decay, on a coarse grid: steady semi-Lagrangian advection."""
        n = land.shape[0]
        src = land * (_smoothstep((0.4 - aridity) / 0.25) + 0.5 * salt_flat + 0.5 * alluvial * _smoothstep((0.6 - aridity) / 0.3))
        if p["loess"] <= 0 or src.max() <= 0:
            return np.zeros((n, n))
        m = min(n, 256)
        z = m / n
        s = ndimage.zoom(src.astype(np.float64), z, order=1)
        vy = ndimage.zoom(data["wind_dy"], z, order=1)
        vx = ndimage.zoom(data["wind_dx"], z, order=1)
        yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
        up = [yy - vy, xx - vx]
        cell_km = ctx.cell_m / z / 1000.0 * ctx.scale
        keep = np.exp(-cell_km / 25.0)                          # dust travels tens of km before it settles
        dust = np.zeros((m, m))
        for _ in range(int(m * 0.8)):
            nd = ndimage.map_coordinates(dust, up, order=1, cval=0.0) * keep + s
            if np.max(np.abs(nd - dust)) < 1e-4:
                dust = nd
                break
            dust = nd
        dust = ndimage.zoom(dust, n / m, order=1)[:n, :n]
        dust = dust / max(float(np.percentile(dust[land], 99)) if land.any() else 1.0, 1e-6)
        # it settles on vegetated land downwind, not where it was lifted
        return np.clip(p["loess"] * 0.6 * dust * (1.0 - _smoothstep((0.45 - aridity) / 0.2)), 0, 1) * land

    @staticmethod
    def _groups(land, depth, sand, clay, om, leach, waterlog, salinity, alluvial, volcanic, aridity, temp, w, slope,
                wet, p):
        """A simplified World Reference Base: the first rule that holds names the soil."""
        s = np.full(land.shape, SOIL_ID["Cambisol"], np.int8)
        old = (w["old"] + w["plain"]) * _smoothstep((0.12 - slope) / 0.08)
        rules = [  # later rules win
            (w["young"] > 0.5, "Regosol"),
            ((old > 0.5) & (clay > 0.3) & (aridity > 0.6), "Luvisol"),
            ((aridity < 0.55), "Kastanozem"),
            ((om > 0.75) & (aridity > 0.45) & (aridity < 1.1), "Chernozem"),
            ((sand > 0.6) & (leach > 0.5) & (temp < 9.0), "Podzol"),
            ((sand > 0.78), "Arenosol"),
            (volcanic, "Andosol"),
            ((alluvial > 0.5), "Fluvisol"),
            ((waterlog > 0.55), "Gleysol"),
            ((wet > 0.5) & (temp < 10.0) & (om > 0.9), "Histosol"),
            ((depth < 0.3), "Leptosol"),
            ((salinity > 0.5), "Solonchak"),
        ]
        for mask, name in rules:
            s[mask] = SOIL_ID[name]
        s[~land] = 0
        return s

    # ------------------------------------------------------------------ detail: tile materials
    def detail(self, ctx, data, p, x_m, y_m):
        h, water, overlays = Hydrology().detail(ctx, data, data["hydrology_params"], x_m, y_m)
        t = tile_ground(ctx, data, p, x_m, y_m, h, water)
        land = h > 0
        colors = {ROCK: (140, 138, 135), GRAVEL: (170, 160, 145), SAND: (225, 210, 160), CLAY: (160, 110, 80)}
        mats = [((t["ground"] == g) & land & ~water, c) for g, c in colors.items()]
        mats.append(((t["ground"] == DIRT) & land & ~water & t["dry"] & ~t["bare"], (190, 180, 105)))         # dry grass
        return h, water, overlays + mats

    # ------------------------------------------------------------------ views
    def _reach(self, ctx, data):
        """Distance (km) from each land cell to the nearest fertile land."""
        land = data["land"] & (data["water_lake_id"] == 0)
        fertile = land & (data["fertility"] >= data["soil_params"]["fertile_min"])
        if not fertile.any():
            return np.where(land, 99.0, 0.0), fertile
        d = ndimage.distance_transform_edt(~fertile) * ctx.cell_m / 1000.0
        return np.where(land, d, 0.0), fertile

    def legend(self, view, ctx, data):
        if view == "soil":
            return [{"color": list(c), "label": n} for n, c in SOILS[1:]]
        if view == "parent":
            pal = [(210, 150, 150), (220, 190, 120), (120, 120, 150), (220, 220, 200), (80, 70, 80),
                   (140, 110, 60), (60, 140, 90), (160, 140, 100), (100, 150, 200), (240, 225, 160), (230, 200, 90)]
            names = list(ROCKS) + ["river gravel and sand", "river silt and clay", "slope colluvium", "lake clay",
                                   "dune sand", "loess"]
            return [{"color": list(c), "label": n} for c, n in zip(pal, names)]
        if view == "texture":
            return [{"color": [255, 0, 0], "label": "red = sand"}, {"color": [0, 255, 0], "label": "green = silt"},
                    {"color": [0, 0, 255], "label": "blue = clay"}]
        if view == "fertility":
            return [{"color": c, "label": l} for c, l in (([150, 120, 95], "0 barren"), ([205, 180, 120], "0.3 poor"),
                    ([215, 215, 120], "0.5 fertile (farming pays)"), ([120, 175, 80], "0.75 very fertile"),
                    ([40, 115, 45], "1 the richest"))]
        if view == "reach":
            return [{"color": [70, 150, 60], "label": "fertile land"},
                    {"color": [240, 240, 200], "label": "infertile, fertile land near (blue) to far (red)"},
                    {"color": [120, 0, 120], "label": "beyond the walking rule"}]
        if view == "detail":
            return [{"color": [140, 138, 135], "label": "rock"}, {"color": [170, 160, 145], "label": "gravel"},
                    {"color": [225, 210, 160], "label": "sand"}, {"color": [160, 110, 80], "label": "clay"},
                    {"color": [190, 180, 105], "label": "dry grass"}, {"color": [55, 115, 200], "label": "water"}]
        return None

    def render(self, view, ctx, data):
        land = data["land"]
        sea = np.array([40, 75, 125], np.uint8)
        if view == "soil":
            img = colormaps.categorical(data["soil_group"], [c for _, c in SOILS])
        elif view == "depth":
            img = colormaps.heat(data["soil_depth_m"], 0.0, 2.5)
        elif view == "texture":
            img = (np.stack([data["sand"], data["silt"], data["clay"]], -1) * 255).clip(0, 255)
        elif view == "parent":
            pal = [(210, 150, 150), (220, 190, 120), (120, 120, 150), (220, 220, 200), (80, 70, 80),
                   (140, 110, 60), (60, 140, 90), (160, 140, 100), (100, 150, 200), (240, 225, 160), (230, 200, 90)]
            img = colormaps.categorical(data["parent"], pal)
        elif view == "reach":
            d, fertile = self._reach(ctx, data)
            limit = data["soil_params"]["walk_min"] / 60.0 * 5.0
            img = colormaps.heat(d, 0.0, limit)
            img[fertile] = (70, 150, 60)
            img[land & (d > limit)] = (120, 0, 120)
        else:
            t = np.clip(data["fertility"], 0, 1)[..., None]
            stops = np.array([0.0, 0.3, 0.5, 0.75, 1.0])
            cols = np.array([(150, 120, 95), (205, 180, 120), (215, 215, 120), (120, 175, 80), (40, 115, 45)], float)
            img = np.stack([np.interp(t[..., 0], stops, cols[:, c]) for c in range(3)], -1)
        img = np.where(land[..., None], img, sea).astype(np.uint8)
        img[data["water_lake_id"] > 0] = (60, 130, 220)
        return img

    def stats(self, ctx, data):
        land = data["land"] & (data["water_lake_id"] == 0)
        if not land.any():
            return {}
        p = data["soil_params"]
        f = data["fertility"][land]
        d, fertile = self._reach(ctx, data)
        limit = p["walk_min"] / 60.0 * 5.0
        infertile = land & ~fertile
        far = infertile & (d > limit)
        groups = np.bincount(data["soil_group"][land], minlength=len(SOILS)) / land.sum() * 100
        top = ", ".join(f"{SOILS[i][0].split(' ')[0]} {groups[i]:.0f}%" for i in np.argsort(-groups)[:5] if groups[i] >= 1)
        return {"fertile land %": round(float((f >= p["fertile_min"]).mean() * 100)),
                "very fertile (≥ 0.75) %": round(float((f >= 0.75).mean() * 100)),
                f"infertile land beyond {p['walk_min']:g} min of fertile %": round(float(far.sum() / max(infertile.sum(), 1) * 100), 1),
                "farthest walk to fertile land (min)": round(float(d[infertile].max() / 5.0 * 60.0)) if infertile.any() else 0,
                "median soil depth (m)": round(float(np.median(data["soil_depth_m"][land])), 2),
                "soils": top}


def tile_ground(ctx, data, p, x_m, y_m, h, water):
    """The ground of each 2 m tile at world points of a regular window (TileGround values: rock,
    gravel, sand, clay, dirt), from the soil and the relief `h` with its water carved in. Also
    returns where the grass would be dry, bare ground, salt, ground within ~6 m of water, and
    the slope. `p` are this stage's parameters. Shared by the detail window and the export."""
    coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
    at = lambda a: ndimage.map_coordinates(np.asarray(a, np.float32), coords, order=1, mode="nearest")
    step = float(x_m[0, 1] - x_m[0, 0])
    gy, gx = np.gradient(h.astype(np.float64), step)
    slope = np.hypot(gx, gy)
    seed = ctx.stage_seed(Soil.id)
    # every threshold below is dithered by noise at two scales, so a limit on a smooth
    # field becomes a ragged mosaic of patches, not one long clean contour line
    n1 = fbm_unit(x_m, y_m, 60, 3, seed + 20)
    n2 = fbm_unit(x_m, y_m, 450, 3, seed + 21)
    depth = at(data["soil_depth_m"]) * (1.0 - _smoothstep((slope - 0.45) / 0.45)) * np.exp(0.25 * n1 + 0.45 * n2)
    sand, clay = at(data["sand"]) + 0.05 * n1, at(data["clay"]) - 0.05 * n1
    coast = at(data["coast_dist_km"])
    ground = np.full(h.shape, DIRT, np.int8)
    # clay shows at the surface where fine sediment settles in water: river banks, lake
    # margins, wet hollows and fine floodplain deposits (where potters find it)
    settles = np.maximum(at(data["waterlog"]), at(data["alluvial"]) * (1.0 - at(data["sand"])))
    # bands a few metres wide along the water (a coarse window has none: one of its cells
    # would widen them to its own size)
    band = lambda m: ndimage.binary_dilation(water, iterations=int(round(m / step))) if step <= m else water
    near_water = band(6.0)
    ground[(clay > 0.33) & ((settles + 0.15 * n1 > 0.45) | (near_water & ~water & (clay > 0.3)))] = CLAY
    ground[(sand > 0.66) | ((coast < 0.08) & (h < 2.5))] = SAND
    # scree below bare rock and gravel bars along fast streams
    grav = (_smoothstep((slope - 0.55) / 0.2) * (0.6 + 0.4 * n1) > 0.5) | \
           (band(4.0) & (at(data["slope"]) > 0.08) & ~water)
    ground[grav] = GRAVEL
    ground[depth < p["rock_depth_m"]] = ROCK
    salt = at(data["salt_flat"]) > 0.5
    dry = at(data["aridity"]) + 0.08 * n2 + 0.04 * n1 < 0.5
    bare = (ground == ROCK) | (ground == GRAVEL) | (ground == SAND) | salt
    return {"ground": ground, "dry": dry, "bare": bare, "salt": salt, "near_water": near_water & ~water,
            "slope": slope}
