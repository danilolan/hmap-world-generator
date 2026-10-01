"""Stage 7 — hydrology (WORLDGEN.md section 2).

Water on the eroded relief, driven by the climate of stage 6:
- runoff: rain minus actual evaporation (Budyko curve in Fu's form), routed down
  the same flow paths as stage 5 into a discharge in m³/s;
- channels start where the discharge passes a threshold, so wet land gets a dense
  network and dry land a sparse one, but only once the catchment is large enough
  for the slope (Montgomery & Dietrich 1988: area x slope² above a constant), so
  steep hills get a dense network and plains a few larger streams instead of
  parallel rills that never meet; each channel's width, depth and speed follow
  its discharge (hydraulic geometry, Leopold & Maddock 1953); brooks, streams and
  unfordable rivers by width; small streams in dry land run only seasonally;
- lakes by water balance: a lake that receives more than it evaporates spills
  through its outlet; one that does not shrinks to the size where evaporation
  equals inflow (a salt lake, its exposed bed a salt flat) or dries completely
  into a salt flat;
- wetlands on floodplains (low height above the nearest channel, HAND) and in
  wet flat hollows (high topographic wetness index), in patches;
- small water the preview grid cannot hold, placed by rule at world coordinates:
  ponds on wet flat ground, tarns in hollows high in young mountains, oxbow lakes
  beside meandering rivers;
- springs (where perennial streams start) and waterfalls (steep drops along a
  channel) as points of interest. A stream that starts on flat ground rises from
  groundwater, so it starts from a spring pool (a small pond at its head) instead of
  out of nothing.
Everything finer than the grid (channel troughs with a water surface that only
runs downhill, valley floors where the micro relief fades, pond bowls) is carved
from world coordinates, so the detail window and the 2 m export agree.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..core import colormaps, hydro
from ..core.lines import chaikin, meander
from ..core.memo import memo
from ..core.params import Float
from ..core.pointnoise import fbm_unit
from ..core.raster_carve import carve_channels, carve_ponds
from ..core.stage import Stage
from .erosion import ground, lake_water, routing_roughness

SECONDS_PER_YEAR = 31_557_600.0
BROOK_MAX_M = 2.5                  # a brook can be stepped or jumped across
BROOK, STREAM, RIVER = 1, 2, 3     # channel classes
POND, TARN, OXBOW, SPRING_POOL = 1, 2, 3, 4    # small water kinds
WATERFALL_MIN_M3S = 0.02           # below this a steep step is a trickle, not a sight
COAST, LAKE, WETLAND, CHANNEL, SMALL_WATER = 1, 2, 3, 4, 5    # what a walk meets


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Hydrology(Stage):
    id = "hydrology"
    title = "Hydrology"
    description = ("Rain becomes runoff and discharge: streams and rivers sized by their flow, lakes that spill or turn "
                   "salt, wetlands, ponds, tarns, oxbow lakes, springs and waterfalls.")
    params = [
        Float("stream_ls", "Streams start at (L/s)", 2.0, 0.5, 100, 0.5,
              "Discharge at which water gathers into a visible stream: lower = denser network. Dry land reaches it "
              "only farther downhill, so it has fewer streams"),
        Float("river_m", "Unfordable rivers from (m wide)", 8.0, 3.0, 30.0, 0.5,
              "Channels at least this wide are rivers (swim, bridge or boat); narrower ones are streams, and brooks "
              f"below {BROOK_MAX_M:g} m"),
        Float("width", "Channel width", 1.0, 0.3, 3.0, 0.05, "Multiplies the width each channel gets from its discharge"),
        Float("seasonal_ls", "Seasonal streams in dry land below (L/s)", 40, 0, 500, 1,
              "In dry land (aridity under 0.5) streams smaller than this flow only in the wet season"),
        Float("evaporation", "Lake evaporation", 1.0, 0.0, 3.0, 0.01,
              "A lake that evaporates more than it receives has no outlet: it shrinks into a salt lake or dries into a "
              "salt flat"),
        Float("wetlands", "Wetlands", 0.5, 0.0, 1.0, 0.01, "Marshes on floodplains and in wet, flat hollows"),
        Float("ponds", "Ponds", 0.5, 0.0, 1.0, 0.01, "Small ponds on wet, flat ground"),
        Float("tarns", "Mountain tarns", 0.5, 0.0, 1.0, 0.01, "Small lakes in hollows high in young mountains"),
        Float("oxbows", "Oxbow lakes", 0.5, 0.0, 1.0, 0.01, "Crescent lakes beside meandering rivers"),
        Float("pond_m", "Pond size (m)", 45, 10, 200, 1, "Typical radius of ponds and tarns"),
        Float("meander", "Meandering", 1.0, 0.0, 3.0, 0.01, "How much channels wind across flat land"),
        Float("waterfall_m", "Waterfalls: smallest drop (m)", 10, 3, 60, 0.5,
              "A steep step along a channel at least this high is a waterfall"),
        Float("channel_km2", "Catchment of a stream on hills (km²)", 0.3, 0.02, 3.0, 0.01,
              "However wet the land, water runs down a hillslope this large (on the map, whatever the world "
              "scale) before it gathers into a stream, on slopes of about 11°; steeper ground needs less (down to "
              "half), flatter ground more (up to 20 times): dense networks in hills, few streams on plains"),
        Float("budyko_w", "Evaporation curve (Fu's w)", 2.6, 1.5, 5.0, 0.05,
              "Budyko curve in Fu's form: higher = more of the rain returns to the air instead of running off",
              advanced=True),
        Float("floodplain_m", "Floodplain height (m)", 5.0, 0.5, 15.0, 0.1,
              "Land less than this above its channel is floodplain", advanced=True),
        Float("pond_spacing_m", "Pond candidate spacing (m)", 220, 80, 800, 5,
              "Spacing of the jittered grid where ponds may appear", advanced=True),
        Float("corridor", "Valley floor width (× channel width)", 4.0, 1.0, 12.0, 0.1,
              "Around each channel, micro relief bumps above the water are cut down into a valley floor reaching this many "
              "channel widths (plus 50 m)", advanced=True),
        Float("flood_ratio", "Floods over the mean flow (×)", 10, 2, 40, 0.5,
              "Channels are cut by their recurring floods (bankfull flow, about the yearly flood), this many times "
              "the mean flow: banks as high and beds as wide as that flood's water (hydraulic geometry). A closed lake "
              "rises to it in wet years, and the ring it then floods is its salt flat", advanced=True),
        Float("min_tributary_m", "Shortest tributary (m)", 150, 0, 600, 10,
              "A tributary shorter than this from its source to the channel it joins is not a channel of its own: "
              "such a short run reaches the river as sheet flow", advanced=True),
        Float("walk_km", "20-minute rule: walk length (km)", 1.7, 0.5, 5.0, 0.05,
              "Straight walks from random land points: how many meet water within this distance", advanced=True),
    ]
    views = {"water": "Water", "discharge": "Discharge", "wetness": "Wetness and floodplains",
             "walk": "Distance to water"}
    units = {'discharge': 'log10 m3/s', 'walk': 'km to water'}
    height_output = "height_eroded"
    has_detail = True

    def run(self, ctx, inputs, p):
        n, dx, S = ctx.res, ctx.cell_m, ctx.scale
        seed = ctx.stage_seed(self.id)
        pe = inputs["erosion_params"]
        h = inputs["height_eroded"].astype(np.float64)
        land = inputs["land"]
        lid5, lvl5, lakes5 = inputs["lake_id"], inputs["lake_level"], inputs["lakes"]
        rain = inputs["rain_mm"].astype(np.float64)
        aridity = inputs["aridity"].astype(np.float64)
        pet = rain / np.maximum(aridity, 1e-4)
        # a cell stands for (dx * scale)² of the uncompressed world: discharges are that world's
        cell_area = (dx * S) ** 2

        # runoff: rain minus actual evaporation, Budyko curve in Fu's form (Fu 1981)
        phi = pet / np.maximum(rain, 1.0)
        w = p["budyko_w"]
        evaporated = 1.0 + phi - (1.0 + phi ** w) ** (1.0 / w)
        runoff = np.clip(rain * (1.0 - evaporated), 0.0, None)                                 # mm/year

        # routing on stage 5's surface (lakes flat at their spill level, same roughness)
        in_lake = lid5 > 0
        rough = routing_roughness(ctx, pe, land) * ~in_lake
        surf = np.where(in_lake, lvl5, h)
        hf, rec, dist, order = hydro.route(surf + rough, land)
        hf = hf - rough.ravel()
        nl = len(lakes5)
        lake_idx = (lid5.ravel() - 1).astype(np.int64)
        pos = np.empty(n * n, np.int64)
        pos[order] = np.arange(order.size)
        outlet = np.full(max(nl, 1), -1, np.int64)
        lake_net = np.zeros(max(nl, 1))
        lc = np.nonzero(lake_idx >= 0)[0]
        if lc.size:
            srt = lc[np.lexsort((pos[lc], lake_idx[lc]))]
            first = srt[np.r_[True, np.diff(lake_idx[srt]) != 0]]
            outlet[lake_idx[first]] = first
            np.add.at(lake_net, lake_idx[lc], (rain.ravel()[lc] - p["evaporation"] * pet.ravel()[lc]) * cell_area / 1000.0)
        local = np.where(land & ~in_lake, runoff, 0.0).ravel() * cell_area / 1000.0          # m³/year
        q, inflow = hydro.discharge(rec, order, local, lake_idx, outlet, lake_net)
        Q = (q / SECONDS_PER_YEAR).reshape(n, n)                                               # m³/s

        lake_map, lake_level, salt, salt_level, lakes = self._lakes(lakes5, lid5, h, rain, pet, inflow, lake_net, cell_area, dx, p)

        # channels and their hydraulic geometry (Leopold & Maddock 1953) from the discharge
        # of the uncompressed world; a channel is walker scale, so its width and depth are
        # never shrunk by the world scale (a larger world has larger rivers)
        # channel heads: area x slope² above a constant (Montgomery & Dietrich 1988), the
        # area measured on the map (walker scale, not the uncompressed world)
        catchment_km2 = hydro.sfd_area(rec, order, dx * dx).reshape(n, n) / 1e6
        gy, gx = np.gradient(ndimage.gaussian_filter(h, 1.0), dx)
        head_km2 = p["channel_km2"] * np.clip((0.2 / np.maximum(np.hypot(gx, gy), 1e-4)) ** 2, 0.5, 20.0)
        channel = land & ~in_lake & (Q * 1000.0 >= p["stream_ls"]) & (catchment_km2 >= head_km2)
        width_v = p["width"] * 7.0 * np.sqrt(Q)
        depth_v = 0.3 * np.power(Q, 0.35)
        width = np.maximum(width_v, 0.5)
        depth = np.maximum(depth_v, 0.1)
        speed = np.where(channel, Q / np.maximum(width_v * depth_v, 1e-6), 0.0)
        cls = np.where(width >= p["river_m"], RIVER, np.where(width >= BROOK_MAX_M, STREAM, BROOK))
        cls = np.where(channel, cls, 0).astype(np.int8)
        seasonal = channel & (aridity < 0.5) & (Q * 1000.0 < p["seasonal_ls"])

        gy, gx = np.gradient(h, dx)
        slope = np.hypot(gx, gy)
        lines, springs, falls = self._trace(channel, seasonal, rec, surf.ravel(), lake_idx, land, Q, width, depth,
                                            speed, slope, n, dx, p, seed)
        segs, seg_cls = self._surface(ctx, inputs, pe, lines, p)

        # wetlands: floodplains (HAND) and wet flat hollows (topographic wetness index)
        hand = hydro.height_above_drainage(surf.ravel(), rec, order, (channel | in_lake).ravel()).reshape(n, n)
        area_mfd = hydro.mfd_area(hf + rough.ravel(), order, land, dx * dx, pe["mfd_exponent"]).reshape(n, n)
        twi = np.log(area_mfd / dx / np.maximum(slope, 1e-3))
        humid = 0.3 + 0.7 * _smoothstep((aridity - 0.5) / 0.5)
        flat = 1.0 - _smoothstep(slope / 0.08)
        flood = 1.0 - _smoothstep(np.maximum(hand, 0) / ctx.height(p["floodplain_m"]))
        wettish = _smoothstep((twi - 9.0) / 4.0)
        yy, xx = np.mgrid[0:n, 0:n]
        mosaic = flat * np.maximum(flood, wettish) * humid + 0.3 * fbm_unit((xx + 0.5) * dx, (yy + 0.5) * dx, 1500, 3, seed + 1)
        wetland = _smoothstep((mosaic - (1.05 - 0.8 * p["wetlands"])) / 0.15)
        wetland = np.where(land & ~in_lake & ~salt, wetland, 0.0).astype(np.float32)

        ponds = self._ponds(ctx, inputs, p, seed, land, in_lake | salt, channel, width, slope, humid, flood, wettish, h,
                            springs)

        return {"discharge_m3s": Q.astype(np.float32), "runoff_mm": runoff.astype(np.float32),
                "channel_class": cls, "seasonal": seasonal, "stream_lines": lines, "stream_segments": segs,
                "stream_segment_class": seg_cls, "water_lake_id": lake_map, "water_lake_level": lake_level,
                "water_lakes": lakes, "salt_flat": salt, "salt_level": salt_level, "wetland": wetland, "hand_m": hand.astype(np.float32),
                "wetness_index": twi.astype(np.float32), "ponds": ponds, "springs": springs, "waterfalls": falls,
                "hydrology_params": dict(p)}

    # ------------------------------------------------------------------ lakes
    @staticmethod
    def _lakes(lakes5, lid5, h, rain, pet, inflow, lake_net, cell_area, dx, p):
        """Stage 5's trapped basins by water balance: open (spilling),         salt lake (shrunk
        until evaporation equals inflow; the exposed bed is a salt flat) or salt flat. The
        salt flat (playa) is the ring a wet year floods: the lake with flood_ratio times its
        inflow, which evaporates again and leaves its salt and mud; the basin's sides above
        it are ordinary land. `salt_level` is the flat's height."""
        n = h.shape[0]
        lake_map = np.zeros((n, n), np.int32)
        lake_level = np.zeros((n, n), np.float32)
        salt = np.zeros((n, n), bool)
        salt_level = np.zeros((n, n), np.float32)
        lakes = []
        for k, L in enumerate(lakes5):
            cells = lid5 == L["id"]
            balance = inflow[k] + lake_net[k]
            level, kind, wet = L["level_m"], "open", cells
            if balance < 0:
                loss = (p["evaporation"] * pet[cells] - rain[cells]).mean() / 1000.0 * cell_area     # m³/year per cell
                keep = int(inflow[k] / loss) if loss > 0 else int(cells.sum())
                beds = np.sort(h[cells])
                if keep < 2:
                    kind, level, wet = "salt flat", float(beds[0]), np.zeros_like(cells)
                else:
                    kind, level = "salt lake", float(beds[min(keep, beds.size - 1)])
                    wet = cells & (h < level)
                keep_wet = int(inflow[k] * p["flood_ratio"] / loss) if loss > 0 else int(cells.sum())
                level_wet = float(beds[min(max(keep_wet, 1), beds.size - 1)])
                flat = cells & ~wet & (h <= max(level_wet, level))
                salt |= flat
                salt_level[flat] = level
            lid = len(lakes) + 1
            fetch = depth = 0.0
            if wet.any():
                lake_map[wet] = lid
                lake_level[wet] = level
                ys, xs = np.nonzero(wet)
                fetch = float(max(np.ptp(xs), np.ptp(ys)) + 1) * dx / 1000.0
                depth = float(level - h[wet].min())
            lakes.append(dict(id=lid, kind=kind, level_m=float(level), area_km2=float(wet.sum() * dx * dx / 1e6),
                              max_depth_m=depth, fetch_km=fetch, inflow_m3s=float(inflow[k] / SECONDS_PER_YEAR),
                              outflow_m3s=float(max(balance, 0.0) / SECONDS_PER_YEAR)))
        return lake_map, lake_level, salt, salt_level, lakes

    # ------------------------------------------------------------------ channels
    @staticmethod
    def _trace(channel, seasonal, rec, zs, lake_idx, land, Q, width, depth, speed, slope, n, dx, p, seed):
        """Channels as smoothed polylines (columns x, y, width, water surface, discharge,
        depth, speed, seasonal), springs and waterfalls. `zs` is the ground per cell
        (lake surface on lakes); the surface column is set by _surface()."""
        cf, lf = channel.ravel(), land.ravel()
        W, D, V, Qf, SE = width.ravel(), depth.ravel(), speed.ravel(), Q.ravel(), seasonal.ravel()
        idx = np.nonzero(cf)[0]
        has_donor = np.zeros(n * n, bool)
        tgt = rec[idx]
        has_donor[tgt[tgt != idx]] = True
        # trace from the sources of the biggest rivers first, so a line runs on until it
        # meets one already traced
        heads = idx[~has_donor[idx]]
        visited = np.zeros(n * n, bool)
        lines, falls, heads_kept, joins = [], [], [], []
        for s in heads:
            chain, c = [], s
            while True:
                chain.append(c)
                r = rec[c]
                if visited[c] or r == c or not lf[r] or lake_idx[r] >= 0:
                    end = r
                    visited[c] = True
                    break
                visited[c] = True
                c = r
            ch = np.array(chain)
            ii, jj = np.divmod(ch, n)
            pts = np.stack([(jj + 0.5) * dx, (ii + 0.5) * dx, W[ch], zs[ch], Qf[ch], D[ch], V[ch],
                            SE[ch].astype(np.float64)], -1)
            if end != ch[-1]:
                ei, ej = divmod(int(end), n)
                last = pts[-1].copy()
                last[0], last[1] = (ej + 0.5) * dx, (ei + 0.5) * dx
                if not lf[end] or lake_idx[end] >= 0:
                    # the mouth's surface is the sea or the lake; a junction with a bigger
                    # channel takes the ground under it, like any other point
                    last[3] = min(max(zs[end], 0.0) if lf[end] else 0.0, last[3])
                pts = np.vstack([pts, last])
            if len(pts) < 2:
                continue
            line_falls = _waterfalls(pts, p["waterfall_m"])
            # bend the grid's straight runs and 45-degree kinks: a displacement field of
            # world position, so lines sharing a junction move it alike
            ox = 0.25 * dx * fbm_unit(pts[:, 0], pts[:, 1], 5 * dx, 3, seed + 21)
            oy = 0.25 * dx * fbm_unit(pts[:, 0], pts[:, 1], 5 * dx, 3, seed + 22)
            if not lf[end] or lake_idx[end] >= 0:
                # a mouth stays on its sea or lake cell: bent onto the shore it would cut a slot
                ox[-1] = oy[-1] = 0.0
            pts[:, 0] += ox
            pts[:, 1] += oy
            lines.append(meander(chaikin(pts, 3), slope, dx, p["meander"], seed + 7 + len(lines)))
            falls.append(line_falls)
            heads_kept.append(s)
            joins.append(bool(lf[end]) and lake_idx[end] < 0 and end != ch[-1])
        keep = _confluences(lines, joins, p["min_tributary_m"])
        springs = [((s % n + 0.5) * dx, (s // n + 0.5) * dx) for s, k in zip(heads_kept, keep) if k and not SE[s]]
        falls = [f for fl, k in zip(falls, keep) if k for f in fl]
        lines = [line for line, k in zip(lines, keep) if k]
        return lines, np.array(springs).reshape(-1, 2), np.array(falls).reshape(-1, 4)

    @staticmethod
    def _surface(ctx, data, pe, lines, p):
        """Water surface along each line and the carving segments. The surface is the
        full ground (macro and micro relief, as the detail window and the export see it)
        under each final point, lowered by the height of the banks, made to
        run only downhill: water never sits above the
        ground under its line, so a channel cuts through bumps and never rides over them
        on an embankment. Segment rows: ax, ay, bx, by, half width, depth, surface at a,
        surface at b, reach of the valley floor, 1 for a line's first segment (nothing is
        carved upslope of a spring)."""
        if not lines:
            return np.zeros((0, 10)), np.zeros(0, np.int8)
        pts = np.concatenate(lines)
        under = ground(ctx, data, pe, pts[:, 0], pts[:, 1]).astype(np.float64)
        segs, seg_cls, k = [], [], 0
        for i, line in enumerate(lines):
            u = under[k:k + len(line)]
            k += len(line)
            u[-1] = min(u[-1], line[-1, 3])                     # the mouth: sea level or the joined channel
            # below the ground by the banks' height: the bed is cut by the recurring flood
            # (flood_ratio times the mean flow), whose water stands deeper than the mean flow's
            # by the hydraulic geometry depth law d ~ Q^0.35; the mouth keeps the level of the
            # sea, lake or channel it joins
            fb = line[:, 5] * bank_rise_factor(p["flood_ratio"])
            fb[-1] = 0.0
            line[:, 3] = np.maximum(np.minimum.accumulate(u - fb), 0.0)
            lines[i] = line.astype(np.float32)
            a, b = line[:-1], line[1:]
            wdt = 0.5 * (a[:, 2] + b[:, 2])
            head = np.zeros(len(a))
            head[0] = 1.0
            segs.append(np.stack([a[:, 0], a[:, 1], b[:, 0], b[:, 1], 0.5 * wdt, 0.5 * (a[:, 5] + b[:, 5]),
                                  a[:, 3], b[:, 3], 0.5 * wdt * p["corridor"] + 50.0, head], -1))
            seg_cls.append(np.where(wdt >= p["river_m"], RIVER, np.where(wdt >= BROOK_MAX_M, STREAM, BROOK)))
        return np.ascontiguousarray(np.concatenate(segs), np.float64), np.concatenate(seg_cls).astype(np.int8)

    # ------------------------------------------------------------------ small water
    @staticmethod
    def _ponds(ctx, data, p, seed, land, taken, channel, width, slope, humid, flood, wettish, h, springs):
        """Ponds, tarns and oxbow lakes on a jittered grid at world coordinates (they do
        not move with the preview resolution). Rows: x, y, a, b, angle, level, depth, kind."""
        n, dx = ctx.res, ctx.cell_m
        sp = p["pond_spacing_m"]
        g = int(ctx.world_m // sp)
        rng = np.random.default_rng(seed + 11)
        jit = rng.random((g, g, 2))
        roll = rng.random((g * g, 3))
        size = np.exp(rng.normal(0, 0.4, g * g))
        ang = rng.uniform(0, np.pi, g * g)
        gy_, gx_ = np.mgrid[0:g, 0:g]
        X = ((gx_ + jit[..., 0]) * sp).ravel()
        Y = ((gy_ + jit[..., 1]) * sp).ravel()
        coords = [Y / dx - 0.5, X / dx - 0.5]
        at = lambda a, o=1: ndimage.map_coordinates(np.asarray(a, np.float32), coords, order=o, mode="nearest")
        is_land = at(land & ~taken, 0) > 0.5
        free = is_land & (at(ndimage.distance_transform_edt(~channel) * dx) - 0.7 * dx > 40.0)
        flat, hum, fl, wt = at(1 - _smoothstep(slope / 0.15)), at(humid), at(flood), at(wettish)
        young = at(data["w_young"])
        hmax = max(float(h[land].max()), 1.0) if land.any() else 1.0
        zero, r = np.zeros_like(X), p["pond_m"] * size
        out = []
        # ponds: wet flat ground
        suit = flat * hum * (0.4 + 0.3 * wt + 0.3 * fl) * (1 - 0.7 * young)
        pond = free & (roll[:, 0] < 2.5 * p["ponds"] * suit)
        out.append(np.stack([X, Y, r, r * 0.75, ang, zero, np.clip(0.08 * r, 0.8, 4.0), zero + POND], -1)[pond])
        # tarns: hollows high in young mountains
        suit = _smoothstep((young - 0.3) / 0.3) * _smoothstep((at(h) / hmax - 0.25) / 0.2) \
            * _smoothstep(at(ndimage.gaussian_filter(h, 2.0) - h) / ctx.height(3.0)) * hum
        tarn = is_land & ~pond & (roll[:, 1] < 1.0 * p["tarns"] * suit)
        out.append(np.stack([X, Y, r, r * 0.85, ang, zero, np.clip(0.15 * r, 2.0, 12.0), zero + TARN], -1)[tarn])
        # oxbow lakes: beside meandering rivers, on their floodplain
        big = channel & (width >= 4.0)
        if big.any():
            db, (bi, bj) = ndimage.distance_transform_edt(~big, return_indices=True)
            wb, db = at(width[bi, bj]), at(db * dx)
            suit = flat * fl * _smoothstep((db - 1.5 * wb) / 20.0) * (1 - _smoothstep((db - 6 * wb - 80) / 40.0))
            ox = is_land & ~pond & ~tarn & (roll[:, 2] < 0.4 * p["oxbows"] * suit)
            # the crescent's inner side faces the river: +v of the pond frame points to it
            ty = (at(bi, 0) + 0.5) * dx - Y
            tx = (at(bj, 0) + 0.5) * dx - X
            a = np.maximum(p["pond_m"], 3.0 * wb) * size
            out.append(np.stack([X, Y, a, a / 3.0, np.arctan2(ty, tx) - np.pi / 2, zero, np.clip(0.3 * wb, 1.0, 4.0),
                                 zero + OXBOW], -1)[ox])
        ponds = np.concatenate(out)
        if not len(ponds):
            return np.zeros((0, 8))
        # settle each pond into the lowest point of the ground nearby (a hollow of the
        # micro relief), then set its level just under the lowest point of its rim; the
        # ground is the same the detail window and the export evaluate. Ponds whose rim
        # is still far from level (on a slope) are dropped
        k = np.linspace(-1.0, 1.0, 9)
        ox, oy = [a.ravel() * 0.6 * sp for a in np.meshgrid(k, k)]
        patch = ground(ctx, data, data["erosion_params"], ponds[:, 0:1] + ox, ponds[:, 1:2] + oy)
        low = np.argmin(patch, axis=1)
        ponds[:, 0] += ox[low]
        ponds[:, 1] += oy[low]
        # spring pools: streams rising on flat ground start from a small pond at their
        # head (groundwater), kept exactly there
        if len(springs):
            si = np.clip((springs[:, 1] / dx).astype(int), 0, n - 1)
            sj = np.clip((springs[:, 0] / dx).astype(int), 0, n - 1)
            flat_head = springs[slope[si, sj] < 0.05]
            ps = 0.6 * p["pond_m"] * np.exp(np.random.default_rng(seed + 13).normal(0, 0.3, len(flat_head)))
            pools = np.stack([flat_head[:, 0], flat_head[:, 1], ps, 0.8 * ps, (flat_head[:, 0] * 0.001) % np.pi,
                              np.zeros(len(ps)), np.clip(0.08 * ps, 0.8, 3.0), np.full(len(ps), SPRING_POOL)], -1)
            ponds = np.concatenate([ponds, pools])
        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ca, sa = np.cos(ponds[:, 4:5]), np.sin(ponds[:, 4:5])
        u, v = 1.3 * ponds[:, 2:3] * np.cos(t), 1.3 * ponds[:, 3:4] * np.sin(t)
        rim = ground(ctx, data, data["erosion_params"], ponds[:, 0:1] + u * ca - v * sa, ponds[:, 1:2] + u * sa + v * ca)
        lo, hi = rim.min(1), rim.max(1)
        # how uneven the rim may be: a pond needs a hollow, not a slope (its bank is cut
        # down to the level, so the high side of an uneven rim becomes a long cut)
        tol = np.where(ponds[:, 7] == TARN, 0.2 * ponds[:, 2] + 5.0,
                       np.where(ponds[:, 7] == SPRING_POOL, 0.25 * ponds[:, 2] + 6.0, 0.1 * ponds[:, 2] + 3.5))
        ponds[:, 5] = lo - 0.3
        return np.ascontiguousarray(ponds[(hi - lo < tol) & (lo > 0.3)], np.float64)

    # ------------------------------------------------------------------ detail
    def detail(self, ctx, data, p, x_m, y_m):
        h = ground(ctx, data, data["erosion_params"], x_m, y_m).astype(np.float64)
        return water_detail(ctx, data, x_m, y_m, h)


    # ------------------------------------------------------------------ views
    @staticmethod
    def _features(ctx, data):
        """Cells where a walk meets water, by kind."""
        f = np.zeros(data["land"].shape, np.int8)
        pd = data["ponds"]
        if len(pd):
            n = ctx.res
            f[np.clip((pd[:, 1] / ctx.cell_m).astype(int), 0, n - 1), np.clip((pd[:, 0] / ctx.cell_m).astype(int), 0, n - 1)] = SMALL_WATER
        f[data["wetland"] > 0.5] = WETLAND
        f[data["channel_class"] > 0] = CHANNEL
        f[data["water_lake_id"] > 0] = LAKE
        f[~data["land"]] = COAST
        return f

    def legend(self, view, ctx, data):
        if view == "detail":
            return [{"color": [55, 115, 200], "label": "water"}, {"color": [100, 130, 80], "label": "wetland"},
                    {"color": [238, 234, 222], "label": "salt flat"}]
        if view == "water":
            return [{"color": [110, 160, 225], "label": "brook (under 2.5 m, step over it)"},
                    {"color": [170, 185, 205], "label": "seasonal brook (dry in summer)"},
                    {"color": [60, 120, 220], "label": "stream"},
                    {"color": [150, 170, 200], "label": "seasonal stream"},
                    {"color": [30, 80, 200], "label": "river (unfordable)"},
                    {"color": [60, 130, 220], "label": "lake"},
                    {"color": [110, 175, 190], "label": "salt lake"},
                    {"color": [238, 234, 222], "label": "salt flat"},
                    {"color": [95, 130, 80], "label": "wetland"},
                    {"color": [70, 140, 235], "label": "pond, tarn, oxbow, spring pool"}]
        if view == "wetness":
            return [{"color": [170, 120, 70], "label": "dry ground"}, {"color": [30, 90, 160], "label": "wet hollows"},
                    {"color": [120, 200, 230], "label": "floodplain (tinted)"}, {"color": [80, 120, 70], "label": "wetland"},
                    {"color": [60, 130, 220], "label": "lake"}]
        return None

    def legend_labels(self, view, ctx, data):
        if view != "water":
            return None
        lakes, cls, dry = data["water_lake_id"], data["channel_class"], data["seasonal"]
        lab = np.full(lakes.shape, -1, np.int16)
        lab[data["wetland"] > 0.5] = 8
        lab[data["salt_flat"]] = 7
        lab[(cls == BROOK) & ~dry] = 0
        lab[(cls == BROOK) & dry] = 1
        lab[(cls == STREAM) & ~dry] = 2
        lab[(cls == STREAM) & dry] = 3
        river = cls == RIVER
        grow = int(round(2 * ctx.res / 1024))
        lab[(ndimage.binary_dilation(river, iterations=grow) & data["land"]) if grow else river] = 4
        salt_ids = [L["id"] for L in data["water_lakes"] if L["kind"] == "salt lake"]
        salt_lake = np.isin(lakes, salt_ids)
        lab[(lakes > 0) & ~salt_lake] = 5
        lab[salt_lake] = 6
        lab[self._features(ctx, data) == SMALL_WATER] = 9
        lab[~data["land"]] = -1
        return lab

    def render(self, view, ctx, data):
        land, h = data["land"], data["height_eroded"]
        sea = np.array([40, 75, 125], np.uint8)
        lakes = data["water_lake_id"]
        if view == "discharge":
            img = colormaps.moisture(np.log10(np.maximum(data["discharge_m3s"], 1e-4)), -3.5, 2.0)
            img[lakes > 0] = (60, 130, 220)
            return np.where(land[..., None], img, sea).astype(np.uint8)
        if view == "wetness":
            img = colormaps.moisture(np.clip((data["wetness_index"] - 4) / 12, 0, 1)).astype(np.float64)
            fp = land & (data["hand_m"] < ctx.height(data["hydrology_params"]["floodplain_m"]))
            img[fp] = 0.6 * img[fp] + 0.4 * np.array([120, 200, 230])
            img[data["wetland"] > 0.5] = (80, 120, 70)
            img[lakes > 0] = (60, 130, 220)
            return np.where(land[..., None], img, sea).astype(np.uint8)
        if view == "walk":
            feat = self._features(ctx, data) > 0
            d = ndimage.distance_transform_edt(~feat) * ctx.cell_m / 1000.0
            img = colormaps.heat(d, 0.0, 1.2)
            img[feat & land] = (60, 130, 220)
            return np.where(land[..., None], img, sea).astype(np.uint8)
        img = colormaps.terrain(h, ctx.cell_m).astype(np.float64)
        wl = data["wetland"] > 0.5
        img[wl] = 0.45 * img[wl] + 0.55 * np.array([95, 130, 80])
        img[data["salt_flat"]] = (238, 234, 222)
        cls, dry = data["channel_class"], data["seasonal"]
        # brooks are much narrower than a preview pixel: half tone, so a dense network
        # reads as fine lines instead of a solid patch
        for m, rgb in (((cls == BROOK) & dry, (170, 185, 205)), ((cls == BROOK) & ~dry, (110, 160, 225))):
            img[m] = 0.5 * img[m] + 0.5 * np.array(rgb)
        img[(cls == STREAM) & dry] = (150, 170, 200)
        img[(cls == STREAM) & ~dry] = (60, 120, 220)
        river = cls == RIVER
        grow = int(round(2 * ctx.res / 1024))
        img[(ndimage.binary_dilation(river, iterations=grow) & land) if grow else river] = (30, 80, 200)
        salt_ids = [L["id"] for L in data["water_lakes"] if L["kind"] == "salt lake"]
        salt_lake = np.isin(lakes, salt_ids)
        img[(lakes > 0) & ~salt_lake] = (60, 130, 220)
        img[salt_lake] = (110, 175, 190)
        img[self._features(ctx, data) == SMALL_WATER] = (70, 140, 235)
        for x, y, _, _ in data["waterfalls"]:
            i, j = int(y / ctx.cell_m), int(x / ctx.cell_m)
            img[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = (255, 255, 255)
        return img.astype(np.uint8)

    def stats(self, ctx, data):
        land = data["land"]
        if not land.any():
            return {}
        seg, sc = data["stream_segments"], data["stream_segment_class"]
        length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) / 1000.0
        km = {c: float(length[sc == c].sum()) for c in (BROOK, STREAM, RIVER)}
        land_km2 = land.sum() * ctx.cell_m ** 2 / 1e6
        kinds = [L["kind"] for L in data["water_lakes"]]
        pk = data["ponds"][:, 7]
        channels = max(int((data["channel_class"] > 0).sum()), 1)
        return {"rivers (km)": round(km[RIVER]), "streams (km)": round(km[STREAM]), "brooks (km)": round(km[BROOK]),
                "channel km per km²": round(sum(km.values()) / land_km2, 2),
                "seasonal channels %": round(float(data["seasonal"].sum() / channels * 100)),
                "largest discharge (m³/s)": round(float(data["discharge_m3s"][land].max()), 1),
                "widest channel (m)": round(float(2 * seg[:, 4].max()), 1) if len(seg) else 0,
                "lakes open / salt / salt flat": f"{kinds.count('open')} / {kinds.count('salt lake')} / {kinds.count('salt flat')}",
                "ponds / tarns / oxbows / spring pools": f"{int((pk == POND).sum())} / {int((pk == TARN).sum())} / "
                                                         f"{int((pk == OXBOW).sum())} / {int((pk == SPRING_POOL).sum())}",
                "wetland % of land": round(float((data["wetland"][land] > 0.5).mean() * 100), 1),
                "springs": len(data["springs"]), "waterfalls": len(data["waterfalls"]), **self._walk(ctx, data)}

    def _walk(self, ctx, data):
        """The 20-minute rule, water only: straight walks in random directions from random
        land points; how far each goes before it meets water of any kind."""
        L = data["hydrology_params"]["walk_km"] * 1000.0
        feat = self._features(ctx, data)
        n, dx = ctx.res, ctx.cell_m
        cand = np.flatnonzero(feat.ravel() == 0)
        if not cand.size:
            return {}
        rng = np.random.default_rng(12345)
        start = rng.choice(cand, 3000)
        y0 = (start // n + rng.random(start.size)) * dx
        x0 = (start % n + rng.random(start.size)) * dx
        th = rng.uniform(0, 2 * np.pi, start.size)
        hit_d = np.full(start.size, np.inf)
        hit_k = np.zeros(start.size, np.int8)
        step = min(dx / 2, 50.0)
        for s in np.arange(step, L + step / 2, step):
            i = np.clip(((y0 + s * np.sin(th)) / dx).astype(int), 0, n - 1)
            j = np.clip(((x0 + s * np.cos(th)) / dx).astype(int), 0, n - 1)
            k = feat[i, j]
            new = (k > 0) & np.isinf(hit_d)
            hit_d[new], hit_k[new] = s, k[new]
        met = np.isfinite(hit_d)
        names = {CHANNEL: "stream", SMALL_WATER: "pond", WETLAND: "wetland", LAKE: "lake", COAST: "coast"}
        first = ", ".join(f"{v} {np.mean(hit_k[met] == k) * 100:.0f}%" for k, v in names.items() if (hit_k == k).any())
        return {f"walks meeting water within {L / 1000:g} km %": round(float(met.mean() * 100)),
                "median walk to water (km)": round(float(np.median(np.where(met, hit_d, 2 * L))) / 1000.0, 2),
                "first water met": first}


def bank_rise_factor(flood_ratio):
    """How much higher than the mean flow's depth the banks stand: the bankfull flood
    (flood_ratio times the mean discharge) is deeper by the hydraulic geometry depth law
    d ~ Q^0.35 (Leopold & Maddock 1953)."""
    return flood_ratio ** 0.35 - 1.0


def _confluences(lines, joins, min_len):
    """Tidy the lines that end by joining another channel (`joins`), in place: the last
    stretch of a tributary running alongside the channel it joins is cut, so it meets it
    at an angle at one point instead of doubling it into a lens-shaped bulge (the grid's
    flow runs side by side for a few cells before merging); a tributary shorter than
    `min_len` from its source to the junction is dropped, as such a short run
    reaches the river as sheet flow, not as a channel of its own. Returns which lines
    to keep."""
    keep = [True] * len(lines)
    if not lines:
        return keep
    pts = np.concatenate([np.asarray(l)[:, :3] for l in lines])
    owner = np.concatenate([np.full(len(l), i) for i, l in enumerate(lines)])
    tree = cKDTree(pts[:, :2])
    for i, (line, j) in enumerate(zip(lines, joins)):
        if not j:
            continue
        a = np.asarray(line)
        # the receiving channel: the nearest other line at the junction point
        cand = tree.query_ball_point(a[-1, :2], 3.0 * a[-1, 2] + 30.0)
        others = [c for c in cand if owner[c] != i]
        if others:
            recv_w = max(pts[c, 2] for c in others)
            cut = len(a) - 1
            while cut > 2:
                r = 0.5 * a[cut - 1, 2] + 0.5 * recv_w + 3.0
                near = [c for c in tree.query_ball_point(a[cut - 1, :2], r) if owner[c] != i]
                if not near:
                    break
                cut -= 1
            if cut < len(a) - 1:
                lines[i] = np.ascontiguousarray(np.vstack([a[:cut], a[-1:]]))
                a = lines[i]
        if np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])).sum() < min_len:
            keep[i] = False
    return keep


def _waterfalls(pts, min_drop):
    """Knickpoints: runs of steps along a channel (rows x, y, width, surface, ...) much
    steeper than the channel just above and below them, whose total drop reaches
    `min_drop` on channels carrying at least WATERFALL_MIN_M3S; one waterfall each, at
    the steepest step (x, y, drop, width). A steep mountain torrent is not a waterfall
    all along; a step in it is."""
    drop = np.maximum(pts[:-1, 3] - pts[1:, 3], 0.0)
    grad = drop / np.maximum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])), 1.0)
    pad = np.pad(grad, 4, mode="edge")
    around = (np.convolve(pad, np.ones(9), "valid") - grad) / 8.0
    steep = (grad > 0.1) & (grad > 1.5 * around) & (pts[:-1, 4] >= WATERFALL_MIN_M3S)
    out, k = [], 0
    while k < len(steep):
        if not steep[k]:
            k += 1
            continue
        e = k
        while e + 1 < len(steep) and steep[e + 1]:
            e += 1
        if drop[k:e + 1].sum() >= min_drop:
            m = k + int(np.argmax(drop[k:e + 1]))
            out.append((0.5 * (pts[m, 0] + pts[m + 1, 0]), 0.5 * (pts[m, 1] + pts[m + 1, 1]),
                        float(drop[k:e + 1].sum()), float(pts[m, 2])))
        k = e + 1
    return out


def water_detail(ctx, data, x_m, y_m, h, extra_segments=None, ponds=None, ids=None):
    """Carve stage 7's water into heights `h` at world points (a regular window): channels
    (plus `extra_segments`, e.g. stage 11's estuaries), ponds and lakes. Returns
    (heights, water mask, overlays for marsh and salt). Shared by every later stage's
    detail window and the 2 m export. With a dict `ids`, also fills in "lake" (lake id
    under each point, 0 none), "pond" (pond row + 1, 0 none) and "marsh"."""
    coords = [y_m / ctx.cell_m - 0.5, x_m / ctx.cell_m - 0.5]
    base = ndimage.map_coordinates(data["height_eroded"], coords, order=3, mode="nearest").astype(np.float64)
    water = np.zeros(h.shape, np.bool_)
    x0, y0, step = float(x_m[0, 0]), float(y_m[0, 0]), float(x_m[0, 1] - x_m[0, 0])
    # salt flats are dead flat (the lake's fine sediment and salt filled the floor): the
    # ground comes down to just above the flat's level, blended in at its edge
    sf = ndimage.map_coordinates(data["salt_flat"].astype(np.float32), coords, order=1, mode="nearest")
    if (sf > 0).any():
        def spread():
            _, (iy, ix) = ndimage.distance_transform_edt(~data["salt_flat"], return_indices=True)
            return data["salt_level"][iy, ix]
        level = ndimage.map_coordinates(memo(data, "salt_level_near", spread), coords, order=0, mode="nearest")
        wgt = np.clip(sf / 0.5, 0, 1)
        wgt = wgt * wgt * (3 - 2 * wgt)
        h = h + wgt * (np.minimum(h, level + 0.25) - h)
    # streams and rivers in their flood-cut beds; estuaries (sea level, no banks of their
    # own) without
    segs = data["stream_segments"]
    if len(segs):
        carve_channels(h, base, x0, y0, step, segs, water, float(data["hydrology_params"]["flood_ratio"]))
    if extra_segments is not None and len(extra_segments):
        carve_channels(h, base, x0, y0, step, np.ascontiguousarray(extra_segments, np.float64), water, 1.0)
    ponds = data["ponds"] if ponds is None else ponds
    pond = np.zeros(h.shape, np.int32)
    if len(ponds):
        carve_ponds(h, x0, y0, step, ponds, water, pond)
    lake = lake_water(data, h, data["water_lake_id"], data["water_lake_level"], data["height_eroded"], coords)
    water |= lake > 0
    salt = ndimage.map_coordinates(data["salt_flat"].astype(np.float32), coords, order=1, mode="nearest") > 0.5
    wf = ndimage.map_coordinates(data["wetland"], coords, order=1, mode="nearest")
    marsh = (wf + 0.25 * fbm_unit(x_m, y_m, 300, 3, ctx.stage_seed(Hydrology.id) + 3) > 0.55) & (h > 0) & ~water
    if ids is not None:
        ids.update(lake=lake, pond=np.where(lake > 0, 0, pond), marsh=marsh, salt=salt & ~water)
    return h.astype(np.float32), water, [(marsh, (100, 130, 80)), (salt & ~water, (238, 234, 222))]
