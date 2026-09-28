"""Stage 3 — terrain character (WORLDGEN.md section 2).

Decides, place by place, what kind of relief the land has, from the plates and
a few regional noises, following how relief forms on Earth:
- young mountains (sharp peaks, deep valleys) along collisions: continent
  against continent (Himalaya, Alps) and ocean under continent (Andes, a range
  along the coast). Pressure sets their strength;
- plateaus just behind the strongest collisions, where thickened crust is
  lifted as a block (Tibet, Altiplano);
- rift valleys where a continent is being pulled apart (East Africa);
- old, worn hill country (rounded serras, rolling hills): old sutures inside the
  continents (Appalachians, Urals) plus broad regions picked by noise;
- plains everywhere else, widest along the coasts and in the basins.
The result is a per-cell mix of weights (young, plateau, old, plain; they sum
to 1 on land) plus the uplift and rift strengths used by the relief stage.

It also lays out the rock: a mosaic of rock types biased by the character
(granite and gneiss in young ranges, basalt on plateaus and volcanic islands,
sandstone, shale and limestone in old country and plains), and how rugged each
region's rock is at walking scale. The micro relief (stage 4) breaks hard,
layered rock into cliff bands, granite into tors, limestone into dolines, and
leaves shale smooth; the soil (stage 8) weathers from the same rocks.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps, raster
from ..core.noise import fbm, warp
from ..core.params import Float
from ..core.stage import Stage
from ..core.pointnoise import fbm_unit
from .land import ISLAND, ISLET
from .plates import CONVERGENT, DIVERGENT, OCEAN

ROCK_TYPES = ("granite", "sandstone", "shale", "limestone", "basalt")
# how readily each rock stands up as crags, ledges and tors (shale weathers smooth)
ROCK_HARDNESS = {"granite": 0.8, "sandstone": 0.7, "shale": 0.0, "limestone": 0.7, "basalt": 0.8}

STYLE_COLORS = {"young": np.array([175, 70, 55]), "plateau": np.array([150, 100, 170]),
                "old": np.array([220, 140, 190]), "plain": np.array([120, 175, 90])}


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Character(Stage):
    id = "character"
    title = "Terrain character"
    description = ("Where the land is young and sharp (ranges along plate collisions), a lifted plateau, a rift, old rounded "
                   "hill country, or plain.")
    params = [
        Float("range_width_km", "Mountain range width (km)", 2.2, 0.8, 6.0, 0.05,
              "Half width of the young ranges along collisions: narrow = compact ranges with abrupt fronts"),
        Float("range_min_pressure", "Weakest collision that builds a range", 0.25, 0.0, 1.0, 0.01,
              "Collisions with less pressure than this leave no range"),
        Float("plateau", "Plateaus behind ranges", 0.5, 0.0, 1.0, 0.01,
              "How often the land just behind a strong collision is lifted into a plateau (0 = never)"),
        Float("old_regions", "Old hill country", 0.45, 0.0, 1.0, 0.01,
              "Share of the remaining land that is old, rounded hill country (serras) instead of plain"),
        Float("old_region_size", "Hill country region size", 3, 1, 10, 0.5,
              "Lower = few large hill regions; higher = many small ones"),
        Float("old_sutures", "Old sutures", 0.8, 0.0, 1.0, 0.01,
              "Worn hill belts along the non-colliding plate borders inside continents (old, eroded ranges)"),
        Float("rift", "Rift valleys", 0.7, 0.0, 1.0, 0.01, "Strength of the valleys where a continent is pulled apart"),
        Float("geology_km", "Rock region size (km)", 4.0, 1.0, 15.0, 0.1,
              "Size of the patches of each rock type (granite, sandstone, shale, limestone, basalt)"),
        Float("rugged", "Rugged country", 0.5, 0.0, 1.0, 0.01,
              "How much of the hard rock breaks into crags, rock walls, tors and sinkholes at walking scale (the rest "
              "stays smooth; shale always weathers smooth)"),
        Float("front_warp_km", "Range edge irregularity (km)", 0.8, 0.0, 3.0, 0.05,
              "How much the edges of ranges wander instead of running straight", advanced=True),
        Float("coastal_plain_km", "Coastal plain width (km)", 1.5, 0.0, 6.0, 0.05,
              "Land near the sea tends to be plain (outside ranges)", advanced=True),
    ]
    views = {"character": "Character", "uplift": "Uplift", "rift": "Rift", "rock": "Rock types",
             "rugged": "Rugged rock"}

    def run(self, ctx, inputs, p):
        seed = ctx.stage_seed(self.id)
        n = ctx.res
        km = 1000.0 / ctx.cell_m
        plate, kind, btype, pressure = inputs["plate"], inputs["plate_kind"], inputs["boundary_type"], inputs["boundary_pressure"]
        land, coast = inputs["land"], inputs["coast_dist_km"]
        lo, hi = raster.boundary_pairs(plate)
        klo, khi = kind[np.maximum(lo, 0)], kind[np.maximum(hi, 0)]
        on = lo >= 0

        rw = ctx.km(p["range_width_km"])
        # --- young ranges: collisions involving continental crust
        coll = on & (btype == CONVERGENT) & ((klo != OCEAN) | (khi != OCEAN)) & (pressure > p["range_min_pressure"])
        strength = np.clip((pressure - p["range_min_pressure"]) / 0.8, 0.3, 1.0)
        uplift = np.zeros((n, n), np.float32)
        if coll.any():
            d, (iy, ix) = ndimage.distance_transform_edt(~coll, return_indices=True)
            d = d / km + ctx.km(p["front_warp_km"]) * fbm(n, 4, 12 * ctx.scale, seed + 1)
            # strength of the nearest collision, blurred: nearest-point lookups jump at
            # the midlines between collisions and would leave straight seams
            s_near = ndimage.gaussian_filter(np.where(coll, strength, 0)[iy, ix].astype(np.float32), ctx.km(2.0) * km)
            along = np.clip(0.6 + 0.4 * fbm(n, 4, 10 * ctx.scale, seed + 2), 0.2, 1.2)            # peaks and saddles
            uplift = (s_near * along * np.exp(-(np.maximum(d, 0) / rw) ** 2)).astype(np.float32)
            # plateaus: a band just behind strong collisions, in patches
            band = _smoothstep((d - 0.8 * rw) / rw) * \
                (1 - _smoothstep((d - 4.0 * rw) / (1.5 * rw)))
            band = ndimage.gaussian_filter(band.astype(np.float32), ctx.km(1.0) * km)
            patches = _smoothstep((fbm(n, 3, 6 * ctx.scale, seed + 3) + 1.0 - 2.0 * p["plateau"]) / 0.6)
            plateau = band * patches * np.clip((s_near - 0.4) / 0.4, 0, 1)
        else:
            plateau = np.zeros((n, n), np.float32)

        # --- rift valleys: divergent borders inside continents
        cont_rift = on & (btype == DIVERGENT) & (klo != OCEAN) & (khi != OCEAN)
        rift = np.zeros((n, n), np.float32)
        if cont_rift.any():
            d = ndimage.distance_transform_edt(~cont_rift) / km
            rift = (p["rift"] * np.exp(-(d / ctx.km(1.5)) ** 2)).astype(np.float32)

        # --- old hill country: old sutures + broad noise regions
        sutures = on & (btype != CONVERGENT) & (klo != OCEAN) & (khi != OCEAN)
        old = np.zeros((n, n), np.float32)
        if sutures.any():
            d = ndimage.distance_transform_edt(~sutures) / km
            old = p["old_sutures"] * np.exp(-(d / ctx.km(4.0)) ** 2)
        rs = p["old_region_size"] * ctx.scale
        regions = warp(fbm(n, 4, rs, seed + 4), n / rs * 0.3, seed + 5, rs)
        old = np.maximum(old, _smoothstep((regions - (1.0 - 2.0 * p["old_regions"])) / 0.7))
        coastal = 1 - _smoothstep(coast / max(ctx.km(p["coastal_plain_km"]), 1e-3))
        old = old * (1 - 0.8 * coastal)

        # --- mix: young first, then plateau, then old, the rest plain
        young = np.clip(uplift / 0.35, 0, 1)
        plat = np.clip(plateau, 0, 1) * (1 - young)
        oldw = np.clip(old, 0, 1) * (1 - young - plat)
        plain = np.clip(1 - young - plat - oldw, 0, 1)
        z = np.where(land, 1.0, 0.0)
        rock_w, rugged = self._geology(ctx, p, inputs, young * z, plat * z, oldw * z, plain * z)
        return {"uplift": uplift * land, "rift": rift * land, "w_young": young * z, "w_plateau": plat * z,
                "w_old": oldw * z, "w_plain": plain * z, "rock_w": rock_w, "rugged": rugged}

    def _geology(self, ctx, p, inputs, young, plateau, old, plain):
        """Rock-type weights (soft, so regions blend over a few hundred metres) and the
        regional ruggedness of the rock at walking scale (0 smooth .. 1 rugged)."""
        n, dx = ctx.res, ctx.cell_m
        seed = ctx.stage_seed(self.id) + 50
        yy, xx = np.mgrid[0:n, 0:n]
        X, Y = (xx + 0.5) * dx, (yy + 0.5) * dx
        volcanic = np.isin(inputs["land_class"], (ISLAND, ISLET))
        bias = {"granite": 1.6 * young + 0.3 * old,
                "sandstone": 0.5 * plain + 0.6 * old + 0.3 * plateau,
                "shale": 0.7 * plain + 0.5 * old,
                "limestone": 0.5 * plain + 0.6 * old + 0.2 * plateau,
                "basalt": 1.4 * plateau + 3.0 * volcanic}
        wl = ctx.km(p["geology_km"] * 1000.0)
        score = np.stack([fbm_unit(X, Y, wl, 3, seed + i) + 2.0 * bias[r] for i, r in enumerate(ROCK_TYPES)])
        e = np.exp((score - score.max(0)) * 3.0)
        rock_w = (e / e.sum(0)).astype(np.float32)
        hard = sum(rock_w[i] * ROCK_HARDNESS[r] for i, r in enumerate(ROCK_TYPES))
        region = fbm_unit(X, Y, wl * 1.5, 3, seed + 10)
        rugged = _smoothstep((region + 1.2 * hard + 0.4 * young - 1.6 + 2.0 * p["rugged"]) / 0.8) * (hard > 0.05)
        return rock_w, rugged.astype(np.float32)

    def render(self, view, ctx, data):
        land = data["land"]
        sea = np.array([40, 75, 125], np.uint8)
        if view == "uplift":
            img = colormaps.heat(data["uplift"], 0, 1)
        elif view == "rift":
            img = colormaps.heat(data["rift"], 0, 1)
        elif view == "rock":
            pal = np.array([(210, 150, 150), (220, 190, 120), (120, 120, 150), (220, 220, 200), (80, 70, 80)], float)
            img = np.tensordot(np.moveaxis(data["rock_w"], 0, -1), pal, axes=1)
        elif view == "rugged":
            img = colormaps.heat(data["rugged"], 0, 1)
        else:
            img = sum(data["w_" + k][..., None] * c for k, c in STYLE_COLORS.items())
        return np.where(land[..., None], img, sea).astype(np.uint8)

    def stats(self, ctx, data):
        land = data["land"]
        area = max(land.sum(), 1)
        out = {k + " %": round(float(data["w_" + k][land].sum() / area * 100), 1) for k in STYLE_COLORS}
        return out
