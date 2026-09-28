"""Stage 12 — 2 m tiles and export (WORLDGEN.md sections 2 and 5).

Everything the game reads comes out of this stage, and all of it is evaluated
point by point from world coordinates with the functions the detail windows of
the earlier stages already use, so what the tool shows is what the export
writes:
- heights: stage 5's ground (macro relief, micro relief, gullies), stage 11's
  shore profiles and stage 7's water carved in;
- the ground of each 2 m tile: stage 8's soil rules, the shore's own
  materials (beach sand and pebbles, rock at cliffs and rocky shores, mud on
  salt marshes and in the water), river and lake beds;
- the cover and its growth: grass or dry grass by biome (stage 9), thinner
  under a closed canopy (stage 10), none on bare rock, scree, dunes, beaches,
  deserts, salt and under water. The table COVER is the base look of each
  biome's ground, edited as data; what grows there (trees, bushes, flowers,
  rocks) is the ecology layer's and the biome profiles' (WORLDGEN.md section 5);
- the 16 m ecology layer and its modifiers (stage 10 at the cell centres).
The writing itself (files, far map, water tables, manifest) is
worldgen/export_job.py, run as its own process from the tool's Export button.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps
from ..core.memo import memo
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.stage import Stage
from . import biomes as B
from . import forests as F
from .coast import Coast
from .soil import CLAY, DIRT, GRAVEL, ROCK, SAND, tile_ground

# TileCover values (TERRAIN.md section 2.1); packed dirt, farmland and paving are made by players
NO_COVER, GRASS, DRY_GRASS = 0, 1, 2
# per biome: the cover and its growth (grass height 0-15) in full light
COVER = {
    B.NONE: (NO_COVER, 0), B.MEADOW: (GRASS, 13), B.BROADLEAF: (GRASS, 10), B.CONIFER: (GRASS, 7),
    B.WETLAND: (GRASS, 15), B.HEATH: (GRASS, 5), B.ALPINE: (GRASS, 5), B.COAST: (GRASS, 9),
    B.ROCK: (NO_COVER, 0), B.MIXED: (GRASS, 9), B.STEPPE: (DRY_GRASS, 9), B.SHRUB: (GRASS, 11),
    B.DESERT: (NO_COVER, 0), B.SALT: (NO_COVER, 0), B.RIPARIAN: (GRASS, 14), B.RAINFOREST: (GRASS, 11),
}
COVER_OF = np.array([COVER[b][0] for b in range(len(B.BIOMES))], np.int8)
GROWTH_OF = np.array([COVER[b][1] for b in range(len(B.BIOMES))], np.float64)
GROUND_NAMES = {DIRT: "dirt", SAND: "sand", CLAY: "clay", GRAVEL: "gravel", ROCK: "rock"}
GROUND_RGB = {DIRT: (120, 95, 65), SAND: (225, 210, 160), CLAY: (160, 110, 80), GRAVEL: (165, 158, 145),
              ROCK: (135, 133, 130)}


def fields_at(ctx, data, x_m, y_m, keys):
    """Preview-grid fields sampled (bilinear) at world points."""
    coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
    return {k: ndimage.map_coordinates(np.asarray(data[k], np.float32), coords, order=1, mode="nearest") for k in keys}


def forest_points(ctx, data, x_m, y_m, dry, sample_m):
    """Stage 10 at world points: canopy (0-15), dominant species, biome and modifiers.
    `dry` says which points are dry land (the rest are water: biome 0, no trees)."""
    f = fields_at(ctx, data, x_m, y_m, B.FIELDS + F.EXTRA)
    f["land"] = dry
    f["lake"] = np.zeros(np.shape(x_m), bool)
    seeds = (ctx.stage_seed(B.Biomes.id), ctx.stage_seed(F.Forests.id))
    return F.forest_at(f, x_m, y_m, data["biome_params"], data["forest_params"], seeds, sample_m)


def tiles(ctx, data, x_m, y_m):
    """Final heights and tile materials on a regular window of world points (rows along
    +y), with the canopy evaluated at the points themselves: the detail window. The
    export takes the canopy from its 16 m ecology cells instead (tiles_ground, then
    tiles_cover). Returns a dict: h (m), water (rivers, lakes, ponds), sea, lake and pond
    ids per point, ground, cover, growth (TerrainTile fields) and biome."""
    base = tiles_ground(ctx, data, x_m, y_m)
    canopy, species, biome, _ = forest_points(ctx, data, x_m, y_m, ~base["under"], float(x_m[0, 1] - x_m[0, 0]))
    return tiles_cover(ctx, data, x_m, y_m, base, canopy, species, biome)


def tiles_ground(ctx, data, x_m, y_m):
    """Heights, water and the ground of each tile: stage 11's shaped shore with stage 7's
    water carved in, stage 8's soil rules, the shore's own materials, and the beds under
    the sea, rivers, lakes and ponds."""
    p = data["export_params"]
    ids = {}
    h, water, _, shore = Coast().shore(ctx, data, data["coast_params"], x_m, y_m, ids)
    h = h.astype(np.float64)
    sea = (h <= 0) & ~water
    under = water | sea
    t = tile_ground(ctx, data, data["soil_params"], x_m, y_m, h, water)
    g = t["ground"].copy()
    n1 = fbm_unit(x_m, y_m, 40, 2, ctx.stage_seed(Export.id) + 1)
    at = lambda a: ndimage.map_coordinates(np.asarray(a, np.float32), [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5],
                                           order=1, mode="nearest")
    # the shore's own materials
    g[shore["sand"]] = SAND
    g[shore["shingle"]] = GRAVEL
    g[shore["rocky"] | shore["cliff"]] = ROCK
    g[shore["marsh"] & (h < 0.6)] = CLAY                                   # mudflats at the tide line
    # the sea floor: sand in the shallows, mud deeper; rock off cliffs and rocky shores
    depth = -h
    sea_floor = np.where(depth < p["sea_sand_m"] * (1.0 + 0.3 * n1), SAND, CLAY)
    shallow = depth < 12.0
    sea_floor = np.where(shallow & (shore["rocky"] | shore["cliff"]), ROCK, sea_floor)
    sea_floor = np.where(shallow & shore["shingle"], GRAVEL, sea_floor)
    sea_floor = np.where(shore["marsh"], CLAY, sea_floor)
    g = np.where(sea, sea_floor, g)
    # river beds: gravel where the water runs fast, sand where it is slow; lake and pond mud
    still = (ids["lake"] > 0) | (ids["pond"] > 0)
    channel = water & ~still
    g = np.where(channel, np.where(at(data["slope"]) + 0.01 * n1 > 0.03, GRAVEL, SAND), g)
    g = np.where(still & (g == DIRT), CLAY, g)
    g = np.where(t["salt"] & ~under, CLAY, g)                              # salt crust on a clay pan
    return {"h": h, "water": water, "sea": sea, "under": under, "lake": ids["lake"], "pond": ids["pond"],
            "ground": g.astype(np.int8), "dry": t["dry"], "salt": t["salt"], "beach": shore["sand"],
            "marsh": shore["marsh"], "n1": n1}


def tiles_cover(ctx, data, x_m, y_m, base, canopy, species, biome=None):
    """The cover and its growth over `base` (tiles_ground's result): the biome's grass
    (COVER), dry where the soil stage says so, thinner under the canopy; none on bare
    ground. `biome` is classified at the points when not given."""
    p = data["export_params"]
    h, g, under, n1 = base["h"], base["ground"], base["under"], base["n1"]
    if biome is None:
        f = fields_at(ctx, data, x_m, y_m, B.FIELDS)
        f["land"] = ~under
        f["lake"] = np.zeros(h.shape, bool)
        biome, _, _ = B.classify(f, x_m, y_m, data["biome_params"], ctx.stage_seed(B.Biomes.id))
    cover = COVER_OF[biome]
    cover = np.where((cover == GRASS) & base["dry"], DRY_GRASS, cover)
    cover = np.where(base["marsh"] & (h >= 0.6), GRASS, cover)
    shade = canopy / 15.0
    growth = GROWTH_OF[biome] * p["grass"] * (1.0 - (1.0 - p["forest_floor"]) * shade) + 1.5 * n1
    growth = np.where(base["marsh"], np.maximum(growth, 11.0), growth)
    # bare ground: rock and scree; loose sand on beaches, dunes, deserts and dry country
    # (sandy soils elsewhere carry grass); clay banks right at the water; needle litter
    # under a closed conifer canopy; salt; anything under water
    bank = ndimage.binary_dilation(base["water"], iterations=1) & ~base["water"]
    loose_sand = (g == SAND) & (base["beach"] | np.isin(biome, (B.COAST, B.DESERT)) | base["dry"] | (h < 2.5))
    litter = (canopy >= 12) & (F.KIND[species] == 2)
    bare = (g == ROCK) | (g == GRAVEL) | loose_sand | ((g == CLAY) & bank) | litter | base["salt"] | under
    cover = np.where(bare, NO_COVER, cover).astype(np.int8)
    growth = np.where(cover > 0, np.clip(np.rint(growth), 0, 15), 0).astype(np.int8)
    return {"h": h.astype(np.float32), "water": base["water"], "sea": base["sea"], "lake": base["lake"],
            "pond": base["pond"], "ground": g, "cover": cover, "growth": growth, "biome": biome}


def tile_raw(ground, cover, growth):
    """TerrainTile.Raw: ground | (cover | growth << 4) << 8."""
    return (ground.astype(np.uint16) | ((cover.astype(np.uint16) | (growth.astype(np.uint16) << 4)) << 8)).astype("<u2")


def tile_rgb(t):
    """Colours of the tile materials: ground where bare, the cover's green or straw
    otherwise (taller grass deeper)."""
    img = np.zeros(t["ground"].shape + (3,))
    for k, c in GROUND_RGB.items():
        img[t["ground"] == k] = c
    gr = t["growth"][..., None] / 15.0
    img = np.where((t["cover"] == GRASS)[..., None], (1 - gr) * np.array([165, 185, 95]) + gr * np.array([70, 135, 50]), img)
    img = np.where((t["cover"] == DRY_GRASS)[..., None], (1 - gr) * np.array([200, 190, 130]) + gr * np.array([185, 165, 90]), img)
    return img


class Export(Stage):
    id = "export"
    title = "2 m tiles and export"
    description = ("The final heights and the material of every 2 m tile (ground, cover, grass height), the 16 m "
                   "ecology layer, water tables and the far map, written in the game's format by Export.")
    params = [
        Float("grass", "Grass height", 1.0, 0.3, 1.3, 0.01,
              "Scales every biome's grass height (the tiles' growth; players' animals graze it down)"),
        Float("forest_floor", "Grass under a closed canopy", 0.35, 0.0, 1.0, 0.01,
              "Share of the grass height kept in deep shade (0: bare litter under every closed canopy)"),
        Float("sea_sand_m", "Sandy sea floor down to (m)", 12, 2, 40, 0.5,
              "Deeper sea floors are mud (clay)", advanced=True),
    ]
    views = {"tiles": "Tile materials"}
    height_output = "height_eroded"
    has_detail = True

    def run(self, ctx, inputs, p):
        return {"export_params": dict(p)}

    def detail(self, ctx, data, p, x_m, y_m):
        data["export_params"] = p
        t = tiles(ctx, data, x_m, y_m)
        img = tile_rgb(t)
        land = ~t["water"] & ~t["sea"]
        mats = [(land & (np.abs(img - c).sum(-1) < 1), tuple(c)) for c in np.unique(img[land].reshape(-1, 3), axis=0)] \
            if land.any() else []
        return t["h"], t["water"], mats

    def _preview(self, ctx, data):
        """Tile materials over the whole map on a 512² grid (a coarse look at the shares)."""
        m = 512
        g = (np.arange(m) + 0.5) * ctx.world_m / m
        X, Y = np.meshgrid(g, g)
        return memo(data, "export_preview", lambda: tiles(ctx, data, X, Y))

    def render(self, view, ctx, data):
        t = self._preview(ctx, data)
        img = tile_rgb(t)
        img[t["sea"] | t["water"]] = (40, 75, 125)
        out = img.astype(np.uint8)
        if out.shape[0] != ctx.res:
            k = ctx.res / out.shape[0]
            out = ndimage.zoom(out, (k, k, 1), order=0)
        return out

    def stats(self, ctx, data):
        t = self._preview(ctx, data)
        land = ~t["sea"] & ~t["water"]
        if not land.any():
            return {}
        g = np.bincount(t["ground"][land], minlength=5) / land.sum() * 100
        c = np.bincount(t["cover"][land], minlength=3) / land.sum() * 100
        return {"ground on land": ", ".join(f"{GROUND_NAMES[k]} {g[k]:.0f}%" for k in (DIRT, SAND, CLAY, GRAVEL, ROCK)),
                "cover": f"grass {c[GRASS]:.0f}%, dry grass {c[DRY_GRASS]:.0f}%, bare {c[NO_COVER]:.0f}%",
                "mean grass height (0-15)": round(float(t["growth"][land & (t["cover"] > 0)].mean()), 1)
                if (land & (t["cover"] > 0)).any() else 0}
