"""Stage 1 — tectonic plates (WORLDGEN.md section 2).

A plate simulation kept as close to real geology as a flat 64 km map allows,
following existing generators (Frozen Fractal's flood-fill plates; Andy
Gainey's Experilous planet generator, 2014; Cortial et al., "Procedural
Tectonic Planets", 2019). Nothing is drawn by hand: the layout comes from the
seed and the knobs.

1. Plates grow from well-spread seeds over small warped Voronoi cells, so
   their sizes and shapes vary. Continental plates grow first, up to a target
   area, away from the map edge; each continent is welded from several
   neighbouring plates (as India and Eurasia). Microcontinents (the big
   islands) grow in open ocean. Oceanic plates fill the rest.
2. Borders are smoothed into long curves (a plate border is a line on a
   sphere, not a staircase of cells), then lightly warped.
3. Each plate moves and rotates about its centre (on Earth, about an Euler
   pole), so relative motion changes along a border. The plates of one
   continent converge, which is what welded them: a collision crosses every
   multi-plate continent and becomes a mountain range in the relief stage.
4. At every border pixel the relative velocity across the border is the
   pressure, as in Experilous: collision, rifting or sliding.
"""
import heapq

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..core import raster
from ..core.params import Bool, Float, Int
from ..core.stage import Stage

OCEAN, CONTINENT, MICRO = 0, 1, 2
CONVERGENT, DIVERGENT, TRANSFORM = 1, 2, 3
BOUNDARY_COLORS = {CONVERGENT: (235, 70, 50), DIVERGENT: (70, 170, 255), TRANSFORM: (200, 200, 200)}


class Plates(Stage):
    id = "plates"
    title = "Tectonic plates"
    description = ("Plates of uneven size and curved borders, moving and rotating. Each continent is welded from colliding "
                   "plates (a mountain range will cross it); big islands sit in open ocean; oceanic plates fill the rest.")
    params = [
        Int("continents", "Continents", 3, 1, 5, help="Number of continents"),
        Float("continent_km2", "Continent size (km²)", 550, 200, 1400, 10,
              "Area of each continent's plates together; the land ends up somewhat smaller"),
        Int("plates_per_continent", "Plates per continent", 2, 1, 4,
            help="Plates welded into each continent. 2 or more: their collision puts a mountain range across the continent"),
        Int("micro", "Big islands", 3, 0, 5, help="Greenland-like islands (microcontinents)"),
        Float("micro_km2", "Big island size (km²)", 70, 15, 300, 5, "Plate area of each big island"),
        Int("ocean_plates", "Oceanic plates", 9, 3, 20, help="More = more ocean borders, so more island arcs and ridges"),
        Int("hotspots", "Hotspots", 2, 0, 6, help="Plumes under the ocean that leave a chain of islands (like Hawaii)"),
        Bool("force_arc", "Guarantee an island arc", True,
             "Make two oceanic plates collide, so a Caribbean-like arc of islands always exists"),
        Float("size_variation", "Size variation", 0.2, 0.0, 0.5, 0.01, "Random spread of plate areas", advanced=True),
        Float("rotation", "Plate rotation", 0.5, 0.0, 1.0, 0.01,
              "How much plates spin: more = border types change more along each border", advanced=True),
        Float("smooth_km", "Border smoothness (km)", 1.5, 0.0, 5.0, 0.05, "Rounds plate borders into long curves", advanced=True),
        Float("warp", "Border warp", 0.35, 0.0, 1.5, 0.01, "Irregularity added to the borders", advanced=True),
        Int("cells", "Border detail (cells)", 220, 60, 500, 10, "Small cells the plates are built from", advanced=True),
        Float("converge_threshold", "Border threshold", 0.2, 0.0, 0.8, 0.01,
              "Relative speed across a border needed to count as collision or rifting instead of sliding", advanced=True),
    ]
    views = {"plates": "Plates", "boundaries": "Borders (pressure)"}

    # ------------------------------------------------------------------ run
    def run(self, ctx, inputs, p):
        seed = ctx.stage_seed(self.id)
        rng = np.random.default_rng(seed)
        n = ctx.res
        W = ctx.world_m / 1000.0                                   # world side in km

        # small cells (jittered grid, resolution independent)
        side = max(4, int(round(np.sqrt(p["cells"]))))
        gy, gx = np.mgrid[0:side, 0:side].astype(np.float64)
        pts = (np.stack([gy.ravel(), gx.ravel()], -1) + 0.5 + rng.uniform(-0.45, 0.45, (side * side, 2))) / side
        cells = raster.warped_voronoi(n, pts * n, p["warp"] * n / side, seed + 10, warp_features=side / 3)
        ncell = len(pts)
        adj = [[] for _ in range(ncell)]
        for a, b in raster.adjacency(cells):
            adj[a].append(b)
            adj[b].append(a)
        cell_km2 = W * W / ncell

        edge_c = np.minimum(pts, 1 - pts).min(axis=1) * W
        border_km = 4.0

        kind, group, target, seeds = [], [], [], []
        owner = np.full(ncell, -1, np.int32)
        allowed = {}

        def grow(plates):
            """Multi-source growth over the cell graph, each plate up to its target area."""
            grown = {k: 0 for k in plates}
            heap = [(0.0, seeds[k], k) for k in plates]
            while heap:
                cost, c, k = heapq.heappop(heap)
                if owner[c] >= 0 or grown[k] >= target[k] or not allowed[k][c]:
                    continue
                owner[c] = k
                grown[k] += 1
                for nb in adj[c]:
                    if owner[nb] < 0 and allowed[k][nb]:
                        heapq.heappush(heap, (cost + np.linalg.norm(pts[nb] - pts[c]) * rng.uniform(0.75, 1.25), nb, k))

        # 1. continents: one seed each, well spread and away from the edge; the other
        #    plates of the same continent start next to it
        K = p["plates_per_continent"]
        plate_km2 = p["continent_km2"] / K
        ok = edge_c > border_km
        dmin = 2.0 * edge_c
        taken = np.zeros(ncell, bool)
        for ci in range(p["continents"]):
            s0 = int(np.argmax(np.where(ok & ~taken, dmin * rng.uniform(0.8, 1.0, ncell), -1)))
            chosen = [s0]
            ang0 = rng.uniform(0, 2 * np.pi)
            for k in range(1, K):
                a_ = ang0 + 2 * np.pi * k / K + rng.uniform(-0.4, 0.4)
                goal = pts[s0] + np.array([np.sin(a_), np.cos(a_)]) * np.sqrt(plate_km2) / W
                dd = np.linalg.norm(pts - goal, axis=1)
                dd[~ok | taken] = np.inf
                dd[chosen] = np.inf
                chosen.append(int(np.argmin(dd)))
            for s_ in chosen:
                taken[s_] = True
                kind.append(CONTINENT)
                group.append(ci)
                target.append(plate_km2 * rng.uniform(1 - p["size_variation"], 1 + p["size_variation"]) / cell_km2)
                seeds.append(s_)
                allowed[len(kind) - 1] = ok
                dmin = np.minimum(dmin, np.linalg.norm(pts - pts[s_], axis=1) * W)
        grow(list(range(len(kind))))

        # 3. microcontinents: in open ocean, clear of the continents and of the strait
        d_land, _ = cKDTree(pts[owner >= 0]).query(pts)
        d_land = d_land * W
        r_micro = np.sqrt(p["micro_km2"] / np.pi)
        cand = (owner < 0) & (edge_c > border_km + 0.6 * r_micro) & (d_land > 2.5 + 0.6 * r_micro)
        picked = []
        for m in range(p["micro"]):
            if not cand.any():
                break
            dist = d_land.copy()
            for s in picked:
                dist = np.minimum(dist, np.linalg.norm(pts - pts[s], axis=1) * W)
            dist[~cand] = -1
            s = int(np.argmax(dist * rng.uniform(0.8, 1.0, ncell)))
            picked.append(s)
            kind.append(MICRO)
            group.append(p["continents"] + m)
            target.append(p["micro_km2"] * rng.uniform(1 - p["size_variation"], 1 + p["size_variation"]) / cell_km2)
            seeds.append(s)
            allowed[len(kind) - 1] = owner < 0
            grow([len(kind) - 1])
            cand &= owner < 0

        # 4. oceanic plates fill the rest
        dmin = np.full(ncell, np.inf)
        for s in seeds:
            dmin = np.minimum(dmin, np.linalg.norm(pts - pts[s], axis=1))
        ocean_ids = []
        for _ in range(p["ocean_plates"]):
            dd = np.where(owner < 0, dmin * rng.uniform(0.8, 1.0, ncell), -1)
            dd[[seeds[k] for k in ocean_ids]] = -1
            if dd.max() <= 0:
                break
            s = int(np.argmax(dd))
            dmin = np.minimum(dmin, np.linalg.norm(pts - pts[s], axis=1))
            kind.append(OCEAN)
            group.append(-1)
            target.append(np.inf)
            seeds.append(s)
            allowed[len(kind) - 1] = np.ones(ncell, bool)
            ocean_ids.append(len(kind) - 1)
        grow(ocean_ids)
        lost = owner < 0
        if lost.any():
            _, near = cKDTree(pts[~lost]).query(pts[lost])
            owner[lost] = owner[~lost][near]
        P = len(kind)
        kind = np.array(kind, np.int8)
        group = np.array(group, np.int16)

        # 5. smooth borders into long curves: blur each plate's mask, keep the strongest
        plate = owner[cells]
        if p["smooth_km"] > 0:
            sig = p["smooth_km"] * n / W
            best = np.full(plate.shape, -1.0, np.float32)
            smoothed = plate.copy()
            for k in range(P):
                m = ndimage.gaussian_filter((plate == k).astype(np.float32), sig)
                upd = m > best
                best[upd] = m[upd]
                smoothed[upd] = k
            plate = smoothed

        # 6. motion: translation + rotation about the centre, then the forced collisions
        with np.errstate(invalid="ignore", divide="ignore"):
            cen = np.nan_to_num(np.array(ndimage.center_of_mass(np.ones_like(plate), plate, range(P))), nan=n / 2) / n
        ang = rng.uniform(0, 2 * np.pi, P)
        vel = np.stack([np.sin(ang), np.cos(ang)], -1) * rng.uniform(0.3, 1.0, P)[:, None]
        omega = rng.uniform(-1, 1, P) * p["rotation"] * 3.0
        pairs_adj = sorted(raster.adjacency(plate))

        def converge(a, b):
            dirn = cen[a] - cen[b]
            vel[b] = vel[a] + 0.9 * dirn / max(np.linalg.norm(dirn), 1e-9)
            omega[b] = omega[a]

        for ci in range(p["continents"]):
            ids = [k for k in range(P) if group[k] == ci]
            for a, b in zip(ids[:-1], ids[1:]):
                converge(a, b)
        if p["force_arc"]:
            oo_pairs = [(a, b) for a, b in pairs_adj if kind[a] == OCEAN and kind[b] == OCEAN]
            if oo_pairs:
                converge(*oo_pairs[int(rng.integers(len(oo_pairs)))])

        # 7. per-pixel pressure across every border (relative velocity . normal)
        lo, hi = raster.boundary_pairs(plate)
        on = lo >= 0
        pressure = np.zeros(plate.shape, np.float32)
        objs = ndimage.find_objects(np.where(on, lo * P + hi + 1, 0))
        for key, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            a, b = divmod(key - 1, P)
            y0, y1 = max(sl[0].start - 4, 0), min(sl[0].stop + 4, n)
            x0, x1 = max(sl[1].start - 4, 0), min(sl[1].stop + 4, n)
            mb = ndimage.gaussian_filter((plate[y0:y1, x0:x1] == b).astype(np.float32), 2.0)
            gy_, gx_ = np.gradient(mb)
            ys, xs = np.nonzero((lo[y0:y1, x0:x1] == a) & (hi[y0:y1, x0:x1] == b))
            nv = np.stack([gy_[ys, xs], gx_[ys, xs]], -1)
            nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
            q = np.stack([(ys + y0 + 0.5) / n, (xs + x0 + 0.5) / n], -1)

            def v_at(k):
                r = q - cen[k]
                return vel[k] + omega[k] * np.stack([-r[:, 1], r[:, 0]], -1)
            pressure[ys + y0, xs + x0] = np.sum((v_at(a) - v_at(b)) * nv, axis=1)
        thr = p["converge_threshold"]
        btype = np.where(~on, 0, np.where(pressure > thr, CONVERGENT,
                                          np.where(pressure < -thr, DIVERGENT, TRANSFORM))).astype(np.int8)

        # 8. hotspots under the ocean, away from land and from the map edge
        yy, xx = np.mgrid[0:n, 0:n]
        far = (ndimage.distance_transform_edt(kind[plate] == OCEAN) * ctx.cell_m > 6000) & ~on
        far &= np.minimum(np.minimum(yy, n - 1 - yy), np.minimum(xx, n - 1 - xx)) * ctx.cell_m > 8000
        cand_px = np.argwhere(far)
        hot = []
        if len(cand_px) and p["hotspots"]:
            for idx in rng.choice(len(cand_px), size=min(p["hotspots"], len(cand_px)), replace=False):
                y, x = cand_px[idx]
                hot.append((y, x, plate[y, x]))

        return {"plate": plate, "plate_kind": kind, "plate_group": group, "plate_vel": vel, "plate_omega": omega,
                "plate_centroid": cen * n, "plate_pairs": np.array(pairs_adj, np.int32).reshape(-1, 2),
                "boundary_type": btype, "boundary_pressure": pressure,
                "hotspots": np.array(hot, np.float64).reshape(-1, 3), "continent_count": p["continents"]}

    # ------------------------------------------------------------------ views
    def legend(self, view, ctx, data):
        if view == "boundaries":
            return [{"color": [255, 60, 40], "label": "collision (brighter = stronger)"},
                    {"color": [60, 160, 255], "label": "rifting"}]
        return [{"color": [110, 150, 80], "label": "continental plate"}, {"color": [150, 150, 90], "label": "microcontinent"},
                {"color": [45, 85, 140], "label": "oceanic plate"},
                {"color": list(BOUNDARY_COLORS[CONVERGENT]), "label": "converging border"},
                {"color": list(BOUNDARY_COLORS[DIVERGENT]), "label": "diverging border"},
                {"color": list(BOUNDARY_COLORS[TRANSFORM]), "label": "sliding border"},
                {"color": [255, 255, 255], "label": "plate motion (arrow)"}, {"color": [255, 200, 0], "label": "hotspot"}]

    def render(self, view, ctx, data):
        plate, kind, btype = data["plate"], data["plate_kind"], data["boundary_type"]
        n = plate.shape[0]
        grow_px = max(2, n // 200)
        thick = ndimage.grey_dilation(btype, size=grow_px)
        if view == "boundaries":
            img = np.full((n, n, 3), 22, np.uint8)
            img[kind[plate] != OCEAN] = (48, 58, 48)
            pr = data["boundary_pressure"]
            mag = ndimage.grey_dilation(np.abs(pr), size=grow_px)
            pos = ndimage.grey_dilation(np.maximum(pr, 0), size=grow_px) >= ndimage.grey_dilation(np.maximum(-pr, 0), size=grow_px)
            t = np.clip(mag / 1.2, 0.2, 1.0)[..., None]
            col = np.where(pos[..., None], np.array([255, 60, 40]), np.array([60, 160, 255])) * t + 30 * (1 - t)
            m = thick > 0
            img[m] = col[m].astype(np.uint8)
            return img
        base = {OCEAN: np.array([45, 85, 140]), CONTINENT: np.array([110, 150, 80]), MICRO: np.array([150, 150, 90])}
        jitter = raster.palette(len(kind), 7, -22, 22).astype(np.int16)
        col = np.array([base[int(k)] for k in kind], np.int16) + jitter
        img = np.clip(col, 0, 255).astype(np.uint8)[plate]
        for t, c in BOUNDARY_COLORS.items():
            img[thick == t] = c
        for (cy, cx), (vy, vx) in zip(data["plate_centroid"], data["plate_vel"]):
            L = n * 0.05
            raster.draw_line(img, cy, cx, cy + vy * L, cx + vx * L, (255, 255, 255), max(1, n // 300))
            img[max(0, int(cy) - 2):int(cy) + 3, max(0, int(cx) - 2):int(cx) + 3] = (0, 0, 0)
        yy, xx = np.ogrid[:n, :n]
        for y, x, _ in data["hotspots"]:
            r = max(3, n // 120)
            img[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = (255, 200, 0)
        return img

    def stats(self, ctx, data):
        kind, group, plate, btype = data["plate_kind"], data["plate_group"], data["plate"], data["boundary_type"]
        km = ctx.cell_m / 1000.0
        lo, hi = raster.boundary_pairs(plate)
        glo, ghi = group[np.maximum(lo, 0)], group[np.maximum(hi, 0)]
        conv = (lo >= 0) & (btype == CONVERGENT)
        out = {"plates": len(kind), "big islands": int((kind == MICRO).sum())}
        out["collision inside each continent (km)"] = ", ".join(
            str(round(float((conv & (glo == ci) & (ghi == ci)).sum() * km / 2), 1)) for ci in range(int(data["continent_count"])))
        oo = conv & (kind[np.maximum(lo, 0)] == OCEAN) & (kind[np.maximum(hi, 0)] == OCEAN)
        out["ocean-ocean collision (km)"] = round(float(oo.sum() * km / 2), 1)
        out["hotspots"] = len(data["hotspots"])
        return out
