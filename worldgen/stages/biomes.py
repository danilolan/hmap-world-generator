"""Stage 9 — biomes (WORLDGEN.md section 2).

The world's look is western and central Europe and New Zealand, the lands of
medieval high fantasy: meadows and oak woods, beech and fir forests, heather
moors, fens, alpine pastures, mossy rainforest on the wettest coasts; a few
exceptions such as dry steppe and sand desert where the climate allows.

Biomes are classified point by point from smooth fields of the earlier stages
(the idea of a Whittaker diagram, temperature against moisture, plus the
treeline, the soil and the water):
- trees grow where the warmest season is warm enough (the treeline follows a
  summer isotherm): above it alpine pasture, then bare rock and scree;
- moisture (the aridity index) sets the potential cover: forest where humid,
  woodland and shrub mosaics where subhumid, steppe where semi-arid, desert
  where arid (sand desert on sandy ground);
- temperature and soil set the forest: broadleaf where mild and fertile,
  mixed where cooler, conifer where cold or on poor acid sand (pine heaths),
  temperate rainforest where it is very wet and mild all year;
- the soil and water override: waterlogged ground is wetland, salt is salt
  flat, thin soil is rock and scree, dunes and beaches are coast, poor acid
  sand under an open sky is heath, floodplains carry riparian woodland;
- the open land of the medieval landscape (meadows, clearings, commons) is a
  mosaic at walking scale, commoner on fertile flats.
Every threshold is dithered by noise at world coordinates, so borders are
ragged mosaics and the same function serves the preview grid, the detail
window and the 16 m ecology layer of the export (WORLDGEN.md section 5). The
stats measure the owner's 20-minute rule with walks that must meet a new biome
or water.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.stage import Stage
from .hydrology import Hydrology

# ids of the ecology layer (WORLDGEN.md section 5): 0-8 agreed with lane E, new ones appended
BIOMES = [
    ("none (water)", (40, 75, 125)),
    ("meadow", (150, 195, 90)),
    ("broadleaf forest", (55, 125, 45)),
    ("conifer forest", (30, 85, 60)),
    ("wetland", (95, 145, 125)),
    ("heath and moor", (150, 115, 140)),
    ("alpine pasture", (185, 205, 150)),
    ("coast and dunes", (230, 215, 165)),
    ("rock and scree", (150, 148, 145)),
    ("mixed forest", (45, 105, 50)),
    ("steppe", (205, 190, 115)),
    ("shrubland", (125, 150, 75)),
    ("sand desert", (235, 205, 135)),
    ("salt flat", (242, 240, 232)),
    ("riparian woodland", (70, 150, 105)),
    ("temperate rainforest", (20, 95, 70)),
]
(NONE, MEADOW, BROADLEAF, CONIFER, WETLAND, HEATH, ALPINE, COAST, ROCK, MIXED, STEPPE, SHRUB, DESERT, SALT, RIPARIAN,
 RAINFOREST) = range(len(BIOMES))
# what each biome is, for the tool's legend and whoever authors its content (trees, bushes,
# grass look, flowers, rocks): the real landscapes it stands for
BIOME_NOTES = {
    MEADOW: "Open grassland, clearings and commons of the medieval landscape (English downs, Bohemian meadows): "
            "tall grass, wildflowers, a lone oak or hawthorn; commoner on fertile flats",
    BROADLEAF: "Mild broadleaf forest: oak, beech, lime, ash (the Bohemian and English lowland woods)",
    CONIFER: "Cold or poor-soil conifer forest: spruce, silver fir, Scots pine on sand (Scandinavian taiga, the "
             "Bohemian Forest's high slopes)",
    WETLAND: "Fens, marshes and bogs: reeds, sedges, alder and willow carr, open water pools",
    HEATH: "Heather moor and heath on poor acid sand or windswept thin uplands (Lüneburg Heath, Scottish moors): "
           "heather, gorse, birch, bracken",
    ALPINE: "Above the treeline: short grass, alpine flowers, dwarf mountain pine near its lower edge (the Alps' "
            "high pastures)",
    COAST: "Beaches and dunes by the sea: marram grass, sand, driftwood",
    ROCK: "Bare rock and scree: high mountains and steep thin-soiled slopes; lichen, a few cushion plants",
    MIXED: "Cool mixed forest: beech with silver fir and spruce (the Carpathian and Alpine montane belt)",
    STEPPE: "Dry grassland where rain is scarce (the Pannonian steppe): dry grass, few trees",
    SHRUB: "Woodland, scrub and grassland mosaic of subhumid country: hawthorn, blackthorn, birch, scattered oaks",
    DESERT: "Sand desert, only where the climate is arid and the ground sandy: dunes, bare sand (a rare exception)",
    SALT: "Salt flat: the dry, dead-flat bed of a salt lake; white crust, no plants",
    RIPARIAN: "Floodplain woods along rivers: alder, willow, ash, poplar; damp, often flooded",
    RAINFOREST: "Temperate rainforest of the wettest mild coasts (New Zealand, the Pacific Northwest): southern "
                "beech, ferns, moss on everything",
}
FOREST_KIND = {BROADLEAF: 1, CONIFER: 2, MIXED: 3, RIPARIAN: 1, RAINFOREST: 3, SHRUB: 1}
FIELDS = ("temp_mean", "temp_winter", "temp_summer", "aridity", "rain_mm", "waterlog", "sand", "soil_depth_m",
          "fertility", "salinity", "alluvial", "slope", "coast_dist_km", "wind_exposure", "wetland", "height_eroded")


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def classify(f, x_m, y_m, p, seed, margins=False):
    """Biome, forest density (0-15) and forest kind at world points. `f` holds the
    smooth fields sampled at those points (FIELDS, plus 'land' and 'lake'). With
    `margins`, also a dict of how far each point is from the limits that shaped it
    (open land, treeline, moisture), in the dithered units the limits use: the forest
    stage builds its edges from them."""
    n1 = fbm_unit(x_m, y_m, 90, 3, seed + 1)              # tens of metres: ragged borders
    n2 = fbm_unit(x_m, y_m, 700, 3, seed + 2)             # hundreds of metres: patches
    n3 = fbm_unit(x_m, y_m, p["open_km"] * 1000.0, 3, seed + 3)   # the open-land mosaic
    d = 0.6 * n1 + 0.4 * n2                               # dither in "noise units"
    ar = f["aridity"] + 0.06 * d
    ts = f["temp_summer"] + 0.8 * d
    tm = f["temp_mean"] + 0.5 * d
    out = np.full(np.shape(x_m), MEADOW, np.int8)

    # potential cover from moisture and warmth
    trees = ts >= p["treeline_c"]
    forest_ok = trees & (ar >= p["forest_aridity"])
    subhumid = trees & (ar >= p["steppe_aridity"]) & (ar < p["forest_aridity"])
    # forest kind: mild and fertile broadleaf, cool mixed, cold conifer; pine on poor acid sand
    kind = np.where(tm >= p["broadleaf_c"], BROADLEAF, np.where(tm >= p["broadleaf_c"] - 3.0, MIXED, CONIFER))
    poor_sand = (f["sand"] + 0.08 * d > 0.62) & (f["fertility"] < 0.45)
    kind = np.where(poor_sand & (kind == BROADLEAF), MIXED, kind)
    kind = np.where(poor_sand & (kind == MIXED), CONIFER, kind)
    rainforest = (f["rain_mm"] * (1.0 + 0.1 * d) > p["rainforest_mm"]) & (f["temp_winter"] > 1.0) & (tm >= 6.0)
    kind = np.where(rainforest, RAINFOREST, kind)
    # the open land of the medieval landscape: a mosaic, commoner on fertile flats
    openness = n3 + 1.2 * (f["fertility"] - 0.5) - 0.8 * _smoothstep((f["slope"] - 0.15) / 0.2) \
        + 2.2 * (p["open_land"] - 0.4)
    is_open = openness + 0.3 * d > 0.35
    out[forest_ok] = kind[forest_ok]
    out[forest_ok & is_open] = MEADOW
    # subhumid: woodland, shrub and grassland mosaic
    sh = subhumid & (n2 + 0.5 * n1 > 0.2)
    out[subhumid] = MEADOW
    out[sh] = SHRUB
    out[subhumid & (n2 + 0.5 * n1 > 0.9)] = kind[subhumid & (n2 + 0.5 * n1 > 0.9)]
    out[trees & (ar < p["steppe_aridity"])] = STEPPE
    desert = ar < p["desert_aridity"]
    out[desert] = STEPPE
    out[desert & (f["sand"] + 0.1 * d > 0.5)] = DESERT
    # above the treeline: alpine pasture, rock higher up
    out[~trees] = ALPINE
    out[(ts < p["treeline_c"] - 5.0) | (~trees & (f["soil_depth_m"] * np.exp(0.4 * d) < 0.2))] = ROCK
    # soil and water overrides
    heath = poor_sand & (ar >= p["forest_aridity"]) & is_open | \
        (f["wind_exposure"] + 0.15 * d > 1.35) & (f["soil_depth_m"] < 0.8) & trees & (ar > 0.8)
    out[heath & (out != ROCK)] = HEATH
    out[(f["soil_depth_m"] * np.exp(0.4 * d) < 0.15) & (f["slope"] > 0.45)] = ROCK
    floodplain = (f["alluvial"] + 0.15 * d > 0.55) & trees & (ar > 0.55)
    out[floodplain & (n2 > -0.2)] = RIPARIAN
    out[f["waterlog"] + 0.15 * d > 0.55] = WETLAND
    out[(f["coast_dist_km"] < 0.12 + 0.05 * d) & (f["height_eroded"] < 6.0) & (f["slope"] < 0.08)] = COAST
    out[f["salinity"] + 0.1 * d > 0.6] = SALT
    out[~f["land"] | f["lake"]] = NONE

    # forest density (canopy cover 0-15) and kind for the ecology layer
    dens = np.zeros(np.shape(x_m))
    forest = np.isin(out, (BROADLEAF, CONIFER, MIXED, RAINFOREST, RIPARIAN))
    dens[forest] = 11.0 + 3.0 * n1[forest] + 2.0 * n2[forest]
    dens[out == SHRUB] = 4.0 + 2.0 * n1[out == SHRUB]
    dens[out == MEADOW] = np.maximum(0.0, 1.5 * n1[out == MEADOW] - 0.5)       # a lone tree here and there
    dens[out == HEATH] = np.maximum(0.0, 2.0 * n1[out == HEATH])
    # thinner at the treeline and on poor, dry ground
    dens *= np.clip((ts - p["treeline_c"]) / 2.0, 0.3, 1.0) * np.clip(0.6 + ar, 0.6, 1.0)
    dens = np.clip(np.round(dens), 0, 15).astype(np.int8)
    fk = np.zeros(np.shape(x_m), np.int8)
    for b, k in FOREST_KIND.items():
        fk[out == b] = k
    fk[out == MEADOW] = np.where(tm[out == MEADOW] >= p["broadleaf_c"], 1, 2)
    fk[dens == 0] = 0
    if margins:
        return out, dens, fk, {"open": openness + 0.3 * d - 0.35, "treeline": ts - p["treeline_c"],
                               "moisture": ar - p["forest_aridity"], "dither": d}
    return out, dens, fk


class Biomes(Stage):
    id = "biomes"
    title = "Biomes"
    description = ("Western Europe and New Zealand: meadows, broadleaf, mixed and conifer forest, heather moor, wetland, "
                   "alpine pasture, riparian woods, mossy rainforest; steppe and sand desert as rare exceptions.")
    params = [
        Float("open_land", "Open land (meadows, clearings)", 0.3, 0.0, 1.0, 0.01,
              "Share of the forest country that is open: meadows, commons, clearings (commoner on fertile flats)"),
        Float("open_km", "Meadow and wood patch size (km)", 1.2, 0.2, 6.0, 0.05,
              "Size of the patches of open land and woods (walking scale, whatever the world scale)"),
        Float("treeline_c", "Treeline: warmest season (°C)", 10.0, 6.0, 14.0, 0.1,
              "Trees grow only where the warm season averages at least this"),
        Float("broadleaf_c", "Broadleaf forest from (°C mean)", 7.5, 2.0, 12.0, 0.1,
              "Warmer than this: oak, beech, lime; 3 °C cooler: mixed with fir and spruce; colder: conifers"),
        Float("rainforest_mm", "Temperate rainforest above (mm/year)", 2200, 1200, 4000, 25,
              "Mild places wetter than this grow mossy, fern-rich rainforest (New Zealand, the Pacific coasts)"),
        Float("forest_aridity", "Forest needs moisture index above", 0.55, 0.4, 1.2, 0.01,
              "Below it (and above the steppe limit) woods break into a shrub and grassland mosaic"),
        Float("steppe_aridity", "Steppe below moisture index", 0.3, 0.15, 0.6, 0.01),
        Float("desert_aridity", "Desert below moisture index", 0.18, 0.05, 0.35, 0.01,
              "Sand desert where the ground is sandy, dry steppe elsewhere"),
        Float("walk_km", "20-minute rule: walk length (km)", 1.7, 0.5, 5.0, 0.05,
              "Straight walks from random land points: how many meet a new biome or water within this distance",
              advanced=True),
    ]
    views = {"biomes": "Biomes", "forest": "Forest density", "variety": "Distance to a change"}
    units = {'forest': 'canopy 0-15', 'variety': 'km'}
    height_output = "height_eroded"
    has_detail = True

    def run(self, ctx, inputs, p):
        n, dx = ctx.res, ctx.cell_m
        f = {k: np.asarray(inputs[k]) for k in FIELDS}
        f["land"] = inputs["land"]
        f["lake"] = inputs["water_lake_id"] > 0
        yy, xx = np.mgrid[0:n, 0:n]
        biome, dens, kind = classify(f, (xx + 0.5) * dx, (yy + 0.5) * dx, p, ctx.stage_seed(self.id))
        return {"biome": biome, "forest_density": dens, "forest_kind": kind, "biome_params": dict(p)}

    def detail(self, ctx, data, p, x_m, y_m):
        h, water, overlays = Hydrology().detail(ctx, data, data["hydrology_params"], x_m, y_m)
        coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
        f = {k: ndimage.map_coordinates(np.asarray(data[k], np.float32), coords, order=1, mode="nearest") for k in FIELDS}
        f["land"] = h > 0
        f["lake"] = water
        biome, dens, _ = classify(f, x_m, y_m, p, ctx.stage_seed(self.id))
        cols = np.array([c for _, c in BIOMES], np.float64)
        shade = 1.0 - 0.03 * dens                       # denser canopy a little darker
        mats = []
        for b in range(1, len(BIOMES)):
            m = (biome == b) & ~water & (h > 0)
            if m.any():
                mats.append((m, tuple(cols[b])))
        # the server blends overlays over the shaded relief; draw the canopy as darker spots
        trees = (dens >= 8) & ~water & (h > 0)
        return h, water, mats + [(trees & (fbm_unit(x_m, y_m, 12, 2, 7) > 0.3), (25, 70, 35))]

    # ------------------------------------------------------------------ views
    def legend(self, view, ctx, data):
        if view in ("biomes", "detail"):
            return [{"color": list(BIOMES[b][1]), "label": BIOMES[b][0], "note": BIOME_NOTES[b]}
                    for b in range(1, len(BIOMES))]
        return None

    def render(self, view, ctx, data):
        land = data["land"]
        if view == "forest":
            d = data["forest_density"].astype(np.float64)
            img = colormaps.moisture(d, 0, 15)
            img[~land] = (40, 75, 125)
            return img.astype(np.uint8)
        if view == "variety":
            dist = self._change_distance(ctx, data)
            img = colormaps.heat(dist, 0.0, data["biome_params"]["walk_km"])
            img[~land] = (40, 75, 125)
            return img.astype(np.uint8)
        img = colormaps.categorical(data["biome"], [c for _, c in BIOMES]).astype(np.float64)
        shade = colormaps.terrain(data["height_eroded"], ctx.cell_m).astype(np.float64)
        lum = shade.mean(-1, keepdims=True) / max(float(shade[land].mean()) if land.any() else 1.0, 1.0)
        img = np.where(land[..., None], img * np.clip(lum, 0.6, 1.3), img)
        cls = data["channel_class"]
        img[cls >= 2] = (60, 120, 220)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _features(data):
        """Cells where a walker meets something new: a biome border, or any water."""
        b = data["biome"]
        edge = np.zeros(b.shape, bool)
        edge[1:, :] |= b[1:, :] != b[:-1, :]
        edge[:-1, :] |= b[1:, :] != b[:-1, :]
        edge[:, 1:] |= b[:, 1:] != b[:, :-1]
        edge[:, :-1] |= b[:, 1:] != b[:, :-1]
        water = ~data["land"] | (data["water_lake_id"] > 0) | (data["channel_class"] > 0) | (data["wetland"] > 0.5)
        return edge, water

    def _change_distance(self, ctx, data):
        edge, water = self._features(data)
        return ndimage.distance_transform_edt(~(edge | water)) * ctx.cell_m / 1000.0

    def stats(self, ctx, data):
        land = data["land"] & (data["water_lake_id"] == 0)
        if not land.any():
            return {}
        share = np.bincount(data["biome"][land], minlength=len(BIOMES)) / land.sum() * 100
        top = ", ".join(f"{BIOMES[i][0]} {share[i]:.0f}%" for i in np.argsort(-share) if share[i] >= 1)
        forest = (data["forest_density"][land] >= 8).mean() * 100
        return {"biomes": top, "forest cover %": round(float(forest)), **self._walk(ctx, data)}

    def _walk(self, ctx, data):
        """The full 20-minute rule: straight walks from random land points; each must meet
        a biome different from its start, or water, within the walk length."""
        L = data["biome_params"]["walk_km"] * 1000.0
        b = data["biome"]
        _, water = self._features(data)
        n, dx = ctx.res, ctx.cell_m
        cand = np.flatnonzero((data["land"] & ~water).ravel())
        if not cand.size:
            return {}
        rng = np.random.default_rng(4321)
        start = rng.choice(cand, 3000)
        y0 = (start // n + rng.random(start.size)) * dx
        x0 = (start % n + rng.random(start.size)) * dx
        th = rng.uniform(0, 2 * np.pi, start.size)
        b0 = b.ravel()[start]
        hit = np.full(start.size, np.inf)
        what = np.zeros(start.size, np.int8)        # 1 water, 2 biome
        step = min(dx / 2, 50.0)
        for s in np.arange(step, L + step / 2, step):
            i = np.clip(((y0 + s * np.sin(th)) / dx).astype(int), 0, n - 1)
            j = np.clip(((x0 + s * np.cos(th)) / dx).astype(int), 0, n - 1)
            w = water[i, j]
            nb = b[i, j] != b0
            new = np.isinf(hit) & (w | nb)
            hit[new] = s
            what[new] = np.where(w[new], 1, 2)
        met = np.isfinite(hit)
        return {f"walks meeting a change within {L / 1000:g} km %": round(float(met.mean() * 100)),
                "median walk to a change (km)": round(float(np.median(np.where(met, hit, 2 * L))) / 1000.0, 2),
                "first change": f"water {np.mean(what[met] == 1) * 100:.0f}%, new biome {np.mean(what[met] == 2) * 100:.0f}%"}
