"""Stage 2 — land and sea, derived from the plates (WORLDGEN.md section 2).

A land score is built and thresholded; nothing is placed by hand:
- continental crust (continental and microcontinent plates), softened and cut
  by domain-warped noise into bays, capes and peninsulas;
- interior basins: low-frequency noise that lowers the crust inside the
  continents; the deep ones flood and become inland seas or, where the coast
  noise opens them to the ocean, Mediterranean-like arms of the sea;
- sea along the midline between different landmasses, so continents and big
  islands stay separate;
- island arcs on the overriding side of every stretch of border where two
  oceanic plates collide (Caribbean, Japan);
- Hawaii-like hotspot chains (hotspot_chains.py): curved tracks, two parallel
  trends, lobed volcanoes that shrink and drown with age, atolls at the old end;
- islets on the shallow water off the coasts; open ocean at the map edge.
Islands below a minimum size sink; small enclosed water becomes land (lakes come
later, from hydrology). Landmasses are classified by where they come from.
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps, raster
from ..core.noise import fbm, warp
from ..core.params import Float, Int
from ..core.stage import Stage
from .hotspot_chains import hotspot_chains
from .plates import BOUNDARY_COLORS, CONVERGENT, OCEAN

SEA, CONT, BIG, ISLAND, ISLET = 0, 1, 2, 3, 4
CLASS_COLORS = [(40, 80, 140), (100, 160, 75), (175, 170, 90), (225, 150, 60), (250, 225, 90)]
CLASS_NAMES = {CONT: "continents", BIG: "big islands", ISLAND: "islands", ISLET: "islets"}


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Land(Stage):
    id = "land"
    title = "Land and sea"
    description = ("Coasts cut by noise into bays and peninsulas; interior basins that flood into inland seas; island arcs "
                   "where oceanic plates collide; island chains behind hotspots; islets off the coasts.")
    params = [
        Float("roughness", "Coast roughness", 1.2, 0.0, 2.5, 0.01, "How strongly bays, capes and peninsulas cut the coasts"),
        Float("coast_features", "Bay and peninsula size", 10, 2, 40, 0.5,
              "Lower = few large bays and peninsulas; higher = many small ones"),
        Float("strait_km", "Sea between landmasses (km)", 3.5, 0.0, 8.0, 0.05,
              "Sea kept between different continents and big islands (it varies along the way)"),
        Float("basins", "Interior basins", 0.5, 0.0, 1.5, 0.01,
              "How deeply basins sink inside the continents: the deep ones become inland seas (Mediterranean-like)"),
        Float("basin_size", "Interior basin size", 5, 2, 14, 0.5, "Lower = few large basins; higher = many small ones"),
        Float("arc_strength", "Island arcs", 1.0, 0.0, 2.0, 0.01, "How much land the island arcs raise (0 = none)"),
        Float("arc_spread_km", "Island arc width (km)", 2.0, 0.3, 6.0, 0.05,
              "Narrow = a single line of islands (Japan); wide = a scattered sea of islands (Caribbean)"),
        Float("islets", "Coastal islets", 0.5, 0.0, 1.5, 0.01, "How many islets dot the shallow water off the coasts"),
        Float("chain_length_km", "Hotspot chain length (km)", 22, 4, 50, 0.5,
              "How far each Hawaii-like chain trails behind its hotspot"),
        Float("young_island_km", "Hotspot: young island size (km)", 1.1, 0.4, 5.0, 0.05,
              "Radius of the newest volcano, at the hotspot; older ones shrink along the chain"),
        Float("aging", "Hotspot: island aging", 0.75, 0.0, 1.2, 0.01,
              "How fast islands erode and sink along the chain; the drowned ones become atolls"),
        Float("atolls", "Hotspot: atolls", 1.0, 0.0, 1.5, 0.01, "Rings of reef islets left by drowned volcanoes (0 = none)"),
        Float("coast_level", "Coast level", 0.0, -0.3, 0.3, 0.005, "Moves every coastline: negative = more land", advanced=True),
        Float("softness_km", "Coast softness (km)", 1.5, 0.3, 6.0, 0.05, "Blur of the crust edge before the noise shapes it", advanced=True),
        Float("coast_warp", "Coast warp", 0.5, 0.0, 1.5, 0.01, "Domain warping of the coast noise (swirling shapes)", advanced=True),
        Float("border_km", "Open ocean at the map edge (km)", 4.0, 0.5, 12.0, 0.1, advanced=True),
        Float("arc_offset_km", "Arc distance from the border (km)", 2.0, 0.0, 8.0, 0.1, advanced=True),
        Float("arc_island_km", "Arc island size (km)", 1.4, 0.3, 5.0, 0.05, advanced=True),
        Float("eruption_spacing_km", "Hotspot: distance between eruptions (km)", 1.0, 0.4, 5.0, 0.05,
              "Average gap between successive volcanoes along a chain", advanced=True),
        Float("rift_arms", "Hotspot: rift arm length", 0.6, 0.0, 1.5, 0.01,
              "How far the rift zones stretch each volcano into lobed, elongated islands", advanced=True),
        Float("min_island_km2", "Smallest island (km²)", 0.02, 0.0, 1.0, 0.005, advanced=True),
        Float("fill_lakes_km2", "Fill enclosed water below (km²)", 3.0, 0.0, 30.0, 0.1, advanced=True),
    ]
    views = {"classes": "Landmasses", "land": "Land / sea", "distance": "Distance to coast", "plates": "Over plate borders"}
    units = {'distance': 'km (negative = sea)'}

    def run(self, ctx, inputs, p):
        seed = ctx.stage_seed(self.id)
        rng = np.random.default_rng(seed + 3)
        n, cell = ctx.res, ctx.cell_m
        km = 1000.0 / cell                                  # cells per km
        plate, kind, group, vel = inputs["plate"], inputs["plate_kind"], inputs["plate_group"], inputs["plate_vel"]
        btype, hot = inputs["boundary_type"], inputs["hotspots"]
        pgroup = group[plate]
        lo, hi = raster.boundary_pairs(plate)

        # --- continental crust, softened and cut by noise
        crust = (kind[plate] != OCEAN).astype(np.float32)
        s = ndimage.gaussian_filter(crust, ctx.km(p["softness_km"]) * km)
        noise = fbm(n, 6, p["coast_features"] * ctx.scale, seed + 1)
        if p["coast_warp"] > 0:
            noise = warp(noise, p["coast_warp"] * n / p["coast_features"], seed + 2, p["coast_features"] / 2)
        score = s + 0.3 * p["roughness"] * noise * (1.0 - 0.6 * _smoothstep((s - 0.5) / 0.5))

        # --- interior basins: sink the inside of the continents in places
        inside = _smoothstep((s - 0.6) / 0.35)
        basin = _smoothstep((fbm(n, 4, p["basin_size"], seed + 30) - 0.3) / 1.2)
        score -= 1.1 * p["basins"] * basin * inside

        # --- sea along the midline between any two other landmasses
        groups = [g for g in np.unique(group) if g >= 0]
        if len(groups) > 1 and p["strait_km"] > 0:
            dists = np.stack([ndimage.distance_transform_edt(pgroup != g) for g in groups]) / km
            dists.sort(axis=0)
            width = p["strait_km"] * np.clip(1.0 + 0.6 * fbm(n, 3, 8, seed + 7), 0.25, 2.0)
            score *= _smoothstep((dists[1] - dists[0]) / (2.0 * width))

        # --- island arcs where oceanic plates collide (overriding side chosen per plate pair)
        blobs = _smoothstep((fbm(n, 3, 64.0 / ctx.km(p["arc_island_km"]), seed + 4) - 0.35) / 0.9)
        oo = (btype == 1) & (kind[np.maximum(lo, 0)] == OCEAN) & (kind[np.maximum(hi, 0)] == OCEAN) & (lo >= 0)
        arcs = np.zeros(plate.shape, np.float32)
        for a, b in {(int(x), int(y)) for x, y in zip(lo[oo], hi[oo])}:
            over = a if rng.random() < 0.5 else b
            d = ndimage.distance_transform_edt(~(oo & (lo == a) & (hi == b))) / km
            band = np.exp(-((d - ctx.km(p["arc_offset_km"])) / ctx.km(p["arc_spread_km"])) ** 2) * (plate == over)
            arcs = np.maximum(arcs, band)
        score += p["arc_strength"] * arcs * blobs

        # --- hotspot chains (Hawaii-like: two trends, lobed volcanoes, ageing, atolls)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        score = np.maximum(score, hotspot_chains(n, km / ctx.scale, hot, plate, vel, inputs["plate_omega"], inputs["plate_centroid"],
                                                 p, rng, seed))

        # --- islets on the shallow water just off the coasts
        shelf = _smoothstep((s - 0.08) / 0.2) * (1 - _smoothstep((s - 0.45) / 0.05))
        score += p["islets"] * shelf * _smoothstep((fbm(n, 3, 48 * ctx.scale, seed + 5) - 1.1) / 0.6)

        # --- the world ends in open ocean (wavy, but always sea at the edge)
        edge_km = np.minimum(np.minimum(yy, n - 1 - yy), np.minimum(xx, n - 1 - xx)) / km
        edge_km = edge_km * np.clip(1.0 + 0.45 * fbm(n, 3, 6, seed + 6), 0.3, 2.0)
        e = _smoothstep(edge_km / max(p["border_km"], 1e-6))
        score = score * e - (1 - e)
        land = score > 0.5 + p["coast_level"]

        # --- cleanup
        cell_km2 = (cell / 1000.0) ** 2
        lab, cnt, areas = raster.component_areas(land, cell_km2)
        if cnt:
            land &= ~np.concatenate([[False], areas < p["min_island_km2"]])[lab]
        wlab, wcnt, wareas = raster.component_areas(~land, cell_km2, structure8=False)
        if wcnt:
            edge_lab = np.unique(np.concatenate([wlab[0], wlab[-1], wlab[:, 0], wlab[:, -1]]))
            fill = np.concatenate([[False], wareas < p["fill_lakes_km2"]])
            fill[edge_lab] = False
            land |= fill[wlab]

        # --- classify landmasses by origin: continent or microcontinent crust, else island/islet
        lab, cnt, areas = raster.component_areas(land, cell_km2)
        n_cont = int(inputs["continent_count"])
        cls_of = np.full(cnt + 1, SEA, np.int8)
        if cnt:
            gid = np.where(pgroup >= 0, pgroup, 99)
            major = ndimage.labeled_comprehension(gid, lab, np.arange(1, cnt + 1), lambda v: np.bincount(v).argmax(), np.int32, 99)
            for i, (a, g) in enumerate(zip(areas, major), start=1):
                if g < n_cont and a > 50:
                    cls_of[i] = CONT
                elif n_cont <= g < 99 and a > 5:
                    cls_of[i] = BIG
                else:
                    cls_of[i] = ISLAND if a >= 1.0 else ISLET
        cls = cls_of[lab]
        cont_main = np.zeros(n_cont)
        if cnt:
            for i, (a, g) in enumerate(zip(areas, major), start=1):
                if g < n_cont:
                    cont_main[g] = max(cont_main[g], a)
        dist_km = (ndimage.distance_transform_edt(land) - ndimage.distance_transform_edt(~land)) / km
        return {"land": land, "land_class": cls, "coast_dist_km": dist_km.astype(np.float32),
                "landmass_areas": np.sort(areas)[::-1] if cnt else np.zeros(0), "continent_main_km2": cont_main}

    def legend(self, view, ctx, data):
        if view in ("classes", "plates"):
            out = [{"color": list(CLASS_COLORS[k]), "label": n} for k, n in CLASS_NAMES.items()]
            out.append({"color": [40, 80, 140], "label": "sea (darker = deeper)"})
            if view == "plates":
                out += [{"color": list(BOUNDARY_COLORS[CONVERGENT]), "label": "colliding plate border"}]
            return out
        if view == "land":
            return [{"color": [235, 235, 235], "label": "land"}, {"color": [25, 25, 25], "label": "sea"}]
        return None

    def legend_labels(self, view, ctx, data):
        land = data["land"]
        if view in ("classes", "plates"):
            return np.where(land, data["land_class"].astype(np.int16) - 1, len(CLASS_NAMES)).astype(np.int16)
        if view == "land":
            return np.where(land, 0, 1).astype(np.int16)
        return None

    def render(self, view, ctx, data):
        land, cls, d = data["land"], data["land_class"], data["coast_dist_km"]
        if view == "land":
            return np.repeat(np.where(land, 235, 25).astype(np.uint8)[..., None], 3, -1)
        if view == "distance":
            return colormaps.heat(d, -10, 10)
        img = colormaps.categorical(cls, CLASS_COLORS)
        depth = np.clip(-d / 12.0, 0, 1)[..., None]
        sea = (np.array([70, 120, 175]) * (1 - depth) + np.array([15, 35, 80]) * depth).astype(np.uint8)
        img = np.where(land[..., None], img, sea)
        if view == "plates":
            thick = ndimage.grey_dilation(data["boundary_type"], size=max(2, land.shape[0] // 200))
            for t, c in BOUNDARY_COLORS.items():
                img[thick == t] = c
        return img

    def stats(self, ctx, data):
        cls, land = data["land_class"], data["land"]
        km = ctx.cell_m / 1000.0
        lab, cnt = ndimage.label(cls > 0, structure=np.ones((3, 3)))
        counts = {k: 0 for k in CLASS_NAMES}
        areas = {k: [] for k in CLASS_NAMES}
        if cnt:
            first = ndimage.labeled_comprehension(cls, lab, np.arange(1, cnt + 1), lambda v: v[0], np.int8, 0)
            sizes = ndimage.sum(np.ones_like(lab, float), lab, np.arange(1, cnt + 1)) * km * km
            for c, a in zip(first, sizes):
                counts[int(c)] += 1
                areas[int(c)].append(int(a))
        n_cont = int(data["continent_count"])
        cross = self._inner_collision_on_land(data, n_cont)
        main = data["continent_main_km2"]
        ok = (bool((main >= 250).all()) and 2 <= counts[BIG] <= 5 and counts[ISLAND] + counts[ISLET] >= 15 and all(cross)
              and max(areas[BIG] or [0]) < 0.35 * float(main.min()))
        return {"land %": round(float(land.mean() * 100), 1),
                "continents km² (main landmass)": ", ".join(str(int(a)) for a in sorted(data["continent_main_km2"], reverse=True)),
                "big islands km²": ", ".join(map(str, sorted(areas[BIG], reverse=True))),
                "islands": counts[ISLAND], "islets": counts[ISLET],
                "range crosses each continent": "yes" if all(cross) else "no",
                "requirements": "met" if ok else "not met"}

    @staticmethod
    def _inner_collision_on_land(data, n_cont):
        plate, group, btype, land = data["plate"], data["plate_group"], data["boundary_type"], data["land"]
        lo, hi = raster.boundary_pairs(plate)
        out = []
        for g in range(n_cont):
            m = (btype == CONVERGENT) & (group[np.maximum(lo, 0)] == g) & (group[np.maximum(hi, 0)] == g) & land
            out.append(bool(m.sum() > 10))
        return out
