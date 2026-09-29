"""Stage 10 — forests and vegetation (WORLDGEN.md section 2).

Refines stage 9's woods into what the ecology layer carries per 16 m cell
(WORLDGEN.md section 5): the canopy density and the dominant tree species.
- Density follows the site: deep, fertile, moist ground grows closed forest;
  thin soil, steep slopes, wind and drought open it up; small gaps
  (windthrow, old trees fallen) break it at the scale of a few trees.
- Species by their niches, as in species distribution models (a suitability
  per species from warmth, moisture, drainage, soil and position, then the most
  suitable wins, with patch noise so stands mix): oak on deep warm soils, beech
  on humid well-drained slopes, lime on warm fertile lowland, ash on moist rich
  valley sides, birch as the pioneer of poor ground and edges, alder and willow
  in wet ground and along rivers, Scots pine on poor sand, spruce in the cold,
  silver fir in cool humid mountains (with beech), southern beech in the
  wettest mild forests (New Zealand), mountain pine as krummholz at the
  treeline, hawthorn in scrub and hedges. A mixed forest is a mosaic of cells
  with different dominants.
- Edges: near the border of the open land the forest thins into a mantle of
  pioneers and thorny scrub (birch, hawthorn), and a belt of scrub reaches out
  into the meadow; at the treeline the forest opens and ends in krummholz.
- Modifiers for the natural objects (the ecology layer's companion file,
  WORLDGEN.md section 5), each 0-15: flower richness (open ground of moderate
  fertility: species-rich meadows are not the richest soils; alpine and steppe
  flowers too), rockiness (thin soil, steep ground, and the foot of cliffs where
  fallen rock gathers), moist undergrowth (ferns, moss, fungi under canopy where
  it is wet, shaded or by water) and dead wood (fallen trunks and stumps, most
  deep inside old, dense forest).
Everything is a function of world coordinates (the same margins and noises as
stage 9), so the preview, the detail window and the export agree.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.stage import Stage
from . import biomes as B
from .hydrology import Hydrology

# dominant tree species: (name, colour, kind 1 broadleaf / 2 conifer)
SPECIES = [
    ("none", (0, 0, 0), 0),
    ("oak", (95, 125, 45), 1),
    ("beech", (60, 140, 55), 1),
    ("lime", (120, 160, 60), 1),
    ("ash", (140, 175, 95), 1),
    ("birch", (185, 200, 120), 1),
    ("alder", (70, 115, 85), 1),
    ("willow", (150, 185, 140), 1),
    ("Scots pine", (65, 95, 60), 2),
    ("spruce", (25, 70, 50), 2),
    ("silver fir", (35, 85, 75), 2),
    ("southern beech", (40, 110, 80), 1),
    ("mountain pine", (95, 110, 80), 2),
    ("hawthorn", (165, 140, 95), 1),
]
(NO_TREE, OAK, BEECH, LIME, ASH, BIRCH, ALDER, WILLOW, PINE, SPRUCE, FIR, SOUTHERN_BEECH, MOUNTAIN_PINE,
 HAWTHORN) = range(len(SPECIES))
KIND = np.array([k for _, _, k in SPECIES], np.int8)
EXTRA = ("northness", "channel_near_m", "cliff_foot")
MODIFIERS = ("flowers", "rockiness", "undergrowth", "dead_wood")     # bits 0-3, 4-7, 8-11, 12-15


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _bell(x, mid, width):
    return np.exp(-((x - mid) / width) ** 2)


def forest_at(f, x_m, y_m, pb, p, seed, sample_m=0.0):
    """Canopy density (0-15) and dominant species at world points. `f` holds stage 9's
    fields plus EXTRA; `pb` are stage 9's parameters. `sample_m` is the spacing of the
    points: noise finer than about two spacings is faded out (on the coarse preview it
    would only alias into speckle; the detail window and the export sample finely)."""
    biome, _, _, mg = B.classify(f, x_m, y_m, pb, seed[0], margins=True)
    s = seed[1]
    keep = lambda wavelength: _smoothstep((wavelength / max(sample_m, 1e-6) - 1.5) / 2.0) if sample_m > 0 else 1.0
    n16 = keep(40) * fbm_unit(x_m, y_m, 40, 2, s + 1)                  # a few trees: gaps, clumps
    n_stand = [keep(p["stand_m"]) * fbm_unit(x_m, y_m, p["stand_m"], 2, s + 10 + k) for k in range(len(SPECIES))]
    tm, ts, ar = f["temp_mean"], f["temp_summer"], f["aridity"]
    wl, sand, fe, depth = f["waterlog"], f["sand"], f["fertility"], f["soil_depth_m"]
    slope, north, rain = f["slope"], f["northness"], f["rain_mm"]
    near_river = 1.0 - _smoothstep(f["channel_near_m"] / 120.0)
    cool = tm - 1.2 * north * _smoothstep(slope / 0.15)       # north slopes are cooler and moister
    drained = 1.0 - wl

    # --- suitability of each species (a species distribution model in miniature)
    suit = np.zeros((len(SPECIES),) + np.shape(x_m))
    suit[OAK] = _bell(cool, 10.0, 2.5) * (0.4 + 0.6 * _smoothstep(depth / 0.8)) * drained * (0.6 + 0.4 * _smoothstep((1.3 - ar) / 0.6))
    suit[BEECH] = _bell(cool, 8.0, 2.2) * _smoothstep((ar - 0.75) / 0.3) * drained ** 2 * (1.0 - 0.5 * sand) * 1.15
    suit[LIME] = _bell(cool, 10.5, 1.5) * _smoothstep((fe - 0.55) / 0.3) * drained * 0.6
    suit[ASH] = _bell(cool, 9.0, 2.5) * _smoothstep((fe - 0.5) / 0.3) * (0.4 + 0.6 * np.maximum(near_river, _smoothstep((ar - 0.9) / 0.4))) * (1.0 - 0.6 * wl) * 0.8
    suit[BIRCH] = 0.15 + 0.45 * _smoothstep((0.45 - fe) / 0.3) + 0.35 * _smoothstep((5.0 - cool) / 3.0)
    suit[ALDER] = _smoothstep((wl - 0.3) / 0.3) * 1.5 + 0.6 * near_river * _smoothstep((ar - 0.6) / 0.3)
    suit[WILLOW] = near_river * _smoothstep((wl - 0.1) / 0.4) * 1.2
    suit[PINE] = _smoothstep((sand - 0.5) / 0.2) * _smoothstep((0.55 - fe) / 0.3) * 1.3 + 0.4 * _smoothstep((0.8 - ar) / 0.4) * _bell(cool, 6.0, 4.0)
    suit[SPRUCE] = _bell(cool, 3.5, 2.2) * _smoothstep((ar - 0.7) / 0.3) * 1.2
    suit[FIR] = _bell(cool, 6.0, 1.8) * _smoothstep((ar - 0.9) / 0.4) * drained
    suit[SOUTHERN_BEECH] = _smoothstep((rain - 0.8 * p["southern_beech_mm"]) / (0.4 * p["southern_beech_mm"])) * _bell(cool, 7.0, 3.0) * 1.4
    suit[MOUNTAIN_PINE] = _smoothstep((1.2 - mg["treeline"]) / 1.2) * 1.6
    suit[HAWTHORN] = 0.3 * _smoothstep((fe - 0.3) / 0.4)
    suit = np.clip(suit, 0, None) * (1.0 + 0.35 * np.stack(n_stand))     # stands: patches of one species

    # --- which species may lead in each biome
    allowed = {
        B.BROADLEAF: (OAK, BEECH, LIME, ASH, BIRCH, ALDER),
        B.MIXED: (BEECH, FIR, SPRUCE, OAK, ASH, BIRCH, PINE),
        B.CONIFER: (SPRUCE, FIR, PINE, BIRCH, MOUNTAIN_PINE),
        B.RAINFOREST: (SOUTHERN_BEECH, FIR, BEECH),
        B.RIPARIAN: (ALDER, WILLOW, ASH, OAK),
        B.SHRUB: (HAWTHORN, BIRCH, OAK, PINE),
        B.MEADOW: (OAK, ASH, LIME, HAWTHORN, BIRCH),
        B.HEATH: (BIRCH, PINE, HAWTHORN),
        B.ALPINE: (MOUNTAIN_PINE, BIRCH),
        B.WETLAND: (ALDER, WILLOW, BIRCH),
        B.STEPPE: (PINE, HAWTHORN, OAK),
    }
    species = np.zeros(np.shape(x_m), np.int8)
    for b, allow in allowed.items():
        m = biome == b
        if not m.any():
            continue
        sub = suit[list(allow)][:, m]
        species[m] = np.array(allow, np.int8)[np.argmax(sub, 0)]

    # --- canopy density: the biome's cover, scaled by the site
    base = np.zeros(np.shape(x_m))
    for b, v in ((B.BROADLEAF, 12.0), (B.MIXED, 12.5), (B.CONIFER, 12.5), (B.RAINFOREST, 14.0), (B.RIPARIAN, 11.0),
                 (B.SHRUB, 5.0), (B.WETLAND, 3.0)):
        base[biome == b] = v
    site = 0.5 * fe + 0.3 * _smoothstep(depth / 0.8) + 0.2 * _smoothstep((ar - 0.4) / 0.6)
    dens = base * np.clip(0.45 + 0.6 * site, 0.4, 1.05)
    dens *= 1.0 - 0.6 * _smoothstep((slope - 0.5) / 0.4)                  # cliffs and very steep slopes
    dens *= 1.0 - 0.3 * _smoothstep((f["wind_exposure"] - 1.2) / 0.4)      # windswept ridges
    # edges: the forest thins toward the open land and the treeline ...
    inside = np.clip(-mg["open"] / p["edge"], 0, 1)
    forest = base >= 11.0
    dens = np.where(forest, dens * (0.45 + 0.55 * _smoothstep(inside)), dens)
    dens = np.where(forest, dens * (0.35 + 0.65 * _smoothstep(mg["treeline"] / 1.5)), dens)
    edge_zone = forest & (inside < 0.3)
    species = np.where(edge_zone & (n_stand[BIRCH] + 0.5 * n16 > 0.3), np.where(fe > 0.5, HAWTHORN, BIRCH), species)
    # ... and a mantle of scrub reaches a little way out into the meadow: a quarter of the
    # edge (tens of metres, like a real woodland mantle; a full edge width, ~120 m, put
    # scrub and trees over nearly half of every meadow)
    # Only meadows that are clearings of forest country (moist enough for forest, open
    # by the open-land mosaic): a subhumid meadow is open for want of rain, its open
    # margin is meaningless (negative), and it once grew closed forest here
    reach = 0.25 * p["edge"]
    mantle = (biome == B.MEADOW) & (mg["open"] >= 0) & (mg["open"] < reach) & (mg["moisture"] >= 0) \
        & (mg["treeline"] > 0)
    mantle_d = 4.0 * np.clip(1.0 - mg["open"] / reach, 0, 1) * (0.6 + 0.4 * n16)
    dens = np.where(mantle, np.maximum(dens, mantle_d), dens)
    species = np.where(mantle & (mantle_d > 1.0), np.where(n16 > 0, HAWTHORN, BIRCH), species)
    # krummholz above the closed forest, lone trees in the open
    krumm = (biome == B.ALPINE) & (mg["treeline"] > -1.0)
    dens = np.where(krumm, np.maximum(dens, 3.0 * (1.0 + mg["treeline"]) * (0.5 + 0.5 * n16)), dens)
    lone = np.isin(biome, (B.MEADOW, B.HEATH, B.STEPPE)) & ~mantle
    dens = np.where(lone, np.maximum(dens, 2.0 * n16 - 0.8), dens)
    # gaps at the scale of a few trees
    dens = dens + np.where(base > 0, 1.5 * n16, 0.0)
    dens = np.clip(np.round(dens), 0, 15).astype(np.int8)
    species = np.where(dens > 0, species, NO_TREE).astype(np.int8)
    # --- modifiers for the natural objects (0-15 each)
    canopy = dens / 15.0
    open_ground = np.isin(biome, (B.MEADOW, B.STEPPE, B.ALPINE, B.HEATH, B.SHRUB, B.COAST)) * (1.0 - canopy)
    flowers = open_ground * (0.35 + 0.65 * _bell(fe, 0.55, 0.3)) * _smoothstep((ar - 0.25) / 0.3) \
        * (1.0 - 0.6 * wl) * (0.75 + 0.35 * n16) + 0.25 * (edge_zone | mantle)
    flowers = np.where(biome == B.ALPINE, np.maximum(flowers, 0.6 * (0.8 + 0.3 * n16)), flowers)
    rock = np.clip(1.0 - depth / 0.8, 0, 1) * 0.6 + 0.5 * _smoothstep((slope - 0.35) / 0.4) + 0.8 * f["cliff_foot"]
    # a few loose stones everywhere, in patches (field stones, erratics)
    rock = rock + 0.1 + 0.15 * np.maximum(n_stand[HAWTHORN], 0)
    rock = np.where(biome == B.ROCK, np.maximum(rock, 0.9), rock) * (0.8 + 0.3 * n16)
    undergrowth = canopy * (0.3 + 0.35 * _smoothstep((ar - 0.8) / 0.6) + 0.2 * _smoothstep(north * _smoothstep(slope / 0.1))
                            + 0.3 * near_river + 0.3 * np.clip(wl, 0, 1)) * (0.8 + 0.3 * n16)
    old = np.clip(inside, 0, 1) * canopy
    dead_wood = old * (0.5 + 0.3 * np.isin(species, (BEECH, SPRUCE, FIR, SOUTHERN_BEECH)) + 0.3 * _smoothstep((ar - 0.8) / 0.6)) \
        * (0.6 + 0.6 * np.maximum(n_stand[0], 0))
    to4 = lambda v: np.clip(np.round(np.nan_to_num(v) * 15.0), 0, 15).astype(np.int8)
    mods = {"flowers": to4(flowers), "rockiness": to4(rock), "undergrowth": to4(undergrowth), "dead_wood": to4(dead_wood)}
    return dens, species, biome, mods


class Forests(Stage):
    id = "forests"
    title = "Forests and vegetation"
    description = ("Canopy density from the site (soil, water, slope, wind), dominant tree species by their niches, "
                   "and forest edges: thinning mantles of birch and hawthorn, krummholz at the treeline.")
    params = [
        Float("edge", "Forest edge width", 0.3, 0.05, 1.0, 0.01,
              "How wide the thinning edge of the woods and the scrub mantle outside them are"),
        Float("stand_m", "Stand size (m)", 180, 40, 1000, 5,
              "Size of the patches where one species leads: smaller = more finely mixed woods"),
        Float("southern_beech_mm", "Southern beech above (mm/year)", 2000, 1000, 4000, 25,
              "The New Zealand beech leads the wettest mild forests"),
    ]
    views = {"species": "Dominant species", "density": "Canopy density", "kind": "Broadleaf / conifer",
             "flowers": "Flower richness", "rockiness": "Rockiness", "undergrowth": "Moist undergrowth",
             "dead_wood": "Dead wood"}
    units = {'density': 'canopy 0-15', 'flowers': '0-15', 'rockiness': '0-15', 'undergrowth': '0-15', 'dead_wood': '0-15'}
    height_output = "height_eroded"
    has_detail = True

    @staticmethod
    def _extra(ctx, data):
        """Fields only this stage needs: slope aspect (1 = facing north) and distance to a channel."""
        h = ndimage.gaussian_filter(data["height_eroded"].astype(np.float64), 1.0)
        gy, gx = np.gradient(h, ctx.cell_m)
        # rows grow southward: ground rising to the south faces north
        north = gy / np.maximum(np.hypot(gx, gy), 1e-6)
        near = ndimage.distance_transform_edt(data["channel_class"] < 2) * ctx.cell_m
        # the foot of cliffs and steep slopes, where fallen rock gathers: gentle ground just
        # below much steeper ground
        slope = np.asarray(data["slope"], np.float64)
        steep_near = ndimage.grey_dilation(slope, size=3)
        foot = _smoothstep((steep_near - 0.6) / 0.4) * (1.0 - _smoothstep((slope - 0.35) / 0.2))
        return {"northness": north.astype(np.float32), "channel_near_m": near.astype(np.float32),
                "cliff_foot": foot.astype(np.float32)}

    def run(self, ctx, inputs, p):
        n, dx = ctx.res, ctx.cell_m
        extra = self._extra(ctx, inputs)
        f = {k: np.asarray(inputs[k]) for k in B.FIELDS}
        f.update(extra)
        f["land"] = inputs["land"]
        f["lake"] = inputs["water_lake_id"] > 0
        yy, xx = np.mgrid[0:n, 0:n]
        seeds = (ctx.stage_seed(B.Biomes.id), ctx.stage_seed(self.id))
        dens, species, _, mods = forest_at(f, (xx + 0.5) * dx, (yy + 0.5) * dx, inputs["biome_params"], p, seeds, dx)
        return {"tree_density": dens, "tree_species": species, "forest_params": dict(p), **extra,
                **{"mod_" + k: v for k, v in mods.items()}}

    def detail(self, ctx, data, p, x_m, y_m):
        h, water, overlays = Hydrology().detail(ctx, data, data["hydrology_params"], x_m, y_m)
        coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
        f = {k: ndimage.map_coordinates(np.asarray(data[k], np.float32), coords, order=1, mode="nearest")
             for k in B.FIELDS + EXTRA}
        f["land"] = h > 0
        f["lake"] = water
        seeds = (ctx.stage_seed(B.Biomes.id), ctx.stage_seed(self.id))
        dens, species, biome, _ = forest_at(f, x_m, y_m, data["biome_params"], p, seeds, float(x_m[0, 1] - x_m[0, 0]))
        ground = []
        for b in range(1, len(B.BIOMES)):
            m = (biome == b) & ~water & (h > 0)
            if m.any():
                ground.append((m, B.BIOMES[b][1]))
        # crowns: a dot pattern whose coverage follows the canopy density, coloured by species
        crown = fbm_unit(x_m, y_m, 10, 2, 11) + 1.6 * (dens / 15.0) - 0.9 > 0
        trees = [((species == s) & crown & ~water & (h > 0), SPECIES[s][1]) for s in range(1, len(SPECIES))]
        return h, water, overlays + ground + [t for t in trees if t[0].any()]

    def legend(self, view, ctx, data):
        if view in ("species", "detail"):
            kinds = {1: "broadleaf", 2: "conifer"}
            return [{"color": list(SPECIES[s][1]), "label": SPECIES[s][0], "note": kinds[SPECIES[s][2]]}
                    for s in range(1, len(SPECIES))] + [{"color": [215, 210, 180], "label": "no trees"}]
        if view == "kind":
            return [{"color": [110, 170, 70], "label": "broadleaf (darker = denser)"},
                    {"color": [30, 85, 60], "label": "conifer (darker = denser)"},
                    {"color": [215, 210, 180], "label": "no trees"}]
        return None

    def render(self, view, ctx, data):
        land = data["land"]
        sea = np.array([40, 75, 125], np.uint8)
        sp, dens = data["tree_species"], data["tree_density"]
        if view in MODIFIERS:
            img = colormaps.heat(data["mod_" + view].astype(np.float64), 0, 15)
        elif view == "density":
            img = colormaps.moisture(dens.astype(np.float64), 0, 15)
        elif view == "kind":
            k = KIND[sp]
            img = np.full(sp.shape + (3,), 200.0)
            img[k == 1] = (110, 170, 70)
            img[k == 2] = (30, 85, 60)
            img *= (0.55 + 0.45 * dens / 15.0)[..., None] ** 0.5
            img[dens == 0] = (215, 210, 180)
        else:
            img = np.array([c for _, c, _ in SPECIES], np.float64)[sp]
            img[dens == 0] = (215, 210, 180)
            img *= (0.7 + 0.3 * dens / 15.0)[..., None]
        img = np.where(land[..., None], img, sea).astype(np.uint8)
        img[data["water_lake_id"] > 0] = (60, 130, 220)
        return img

    def stats(self, ctx, data):
        land = data["land"] & (data["water_lake_id"] == 0)
        if not land.any():
            return {}
        sp, dens = data["tree_species"][land], data["tree_density"][land]
        treed = dens > 0
        share = np.bincount(sp[dens >= 8], minlength=len(SPECIES)) / max((dens >= 8).sum(), 1) * 100
        top = ", ".join(f"{SPECIES[i][0]} {share[i]:.0f}%" for i in np.argsort(-share) if i and share[i] >= 1)
        return {"closed forest (canopy ≥ 8) %": round(float((dens >= 8).mean() * 100)),
                "any trees %": round(float(treed.mean() * 100)),
                "open woodland and scrub (1-7) %": round(float(((dens > 0) & (dens < 8)).mean() * 100)),
                "species in closed forest": top,
                "modifiers, mean on land (0-15)": ", ".join(f"{k} {data['mod_' + k][land].mean():.1f}" for k in MODIFIERS)}
