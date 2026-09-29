"""The 2 m export (WORLDGEN.md section 5), run as its own process:

    .venv\\Scripts\\python -m worldgen.export_job <job.json>

job.json: seed, res (the grid the pipeline runs on, 1024 or more), scale,
vscale, params (every stage's, as the tool sends them), map_id, full_dir (the
full-resolution files, outside the repository), package_dir (the client
package under Assets/StreamingAssets/World/), workers; optionally files, a list
of [fx, fz] to write only those (tests).

The pipeline runs once; its outputs are saved to a work folder and memory-mapped
by a pool of worker processes, each writing whole 1600² files. A file is made
of four 800² blocks, each evaluated with a margin of 64 corners (the shore
profiles, the waterline tidy and the smoothing near the coast look that far),
so neighbouring blocks agree. Progress goes to <full_dir>/export_status.json.

Axes: the generator's y grows southward (row 0 of its maps is the north); the
game's z grows northward, so world z (m) = world side - y.
"""
import hashlib
import json
import multiprocessing as mp
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .core.pipeline import Pipeline
from .stages import STAGES
from .stages import biomes as B
from .stages import export as X
from .stages import forests as F

GENERATOR_VERSION = "1.0.0"
CORNERS = 32000
STEP_M = 2.0
FILE = 1600
FILES = CORNERS // FILE
BLOCK = 800
MARGIN = 64                      # corners, a multiple of the ecology stride
HEIGHT_STRIDE, TILE_STRIDE, ECO = 4, 8, 8
SEA_LEVEL_RAW = 10000
UNIT_M, ORIGIN_M = 0.1, -1000.0
ECOLOGY_VERSION = 1
SOIL_UNIT_M = 0.1                # soil depth files: u8 per corner, 0.1 m (0-25.5 m)


def status(path, **kw):
    """Write the progress file. On Windows the replace fails while a reader (the tool,
    another session) has the file open: retry briefly, then skip this update; progress
    reports never stop the export."""
    kw["time"] = time.time()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(kw), encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)


# ---------------------------------------------------------------------- pipeline data on disk
def save_data(ctx, data, work):
    work.mkdir(parents=True, exist_ok=True)
    rest = {}
    for k, v in data.items():
        if k == "_memo":                     # rebuilt per worker
            continue
        if isinstance(v, np.ndarray) and v.dtype != object:
            np.save(work / f"{k}.npy", v)
        else:
            rest[k] = v
    with open(work / "rest.pkl", "wb") as f:
        pickle.dump((ctx, rest), f)


def load_data(work):
    with open(work / "rest.pkl", "rb") as f:
        ctx, data = pickle.load(f)
    for npy in work.glob("*.npy"):
        data[npy.stem] = np.load(npy, mmap_mode="r")
    return ctx, data


# ---------------------------------------------------------------------- workers
_W = {}


def _init(work, out):
    _W["ctx"], _W["data"] = load_data(Path(work))
    _W["out"] = out


def _open(name, shape):
    return np.memmap(_W["out"][name], dtype="<u2", mode="r+", shape=shape)


def block(ctx, data, X0, Z0, n=BLOCK, margin=MARGIN):
    """Heights, tiles, ecology and still-water ids for corners [X0, X0+n) x [Z0, Z0+n),
    rows from the south (the game's order)."""
    N = n + 2 * margin
    world = ctx.world_m
    xs = (X0 - margin + np.arange(N)) * STEP_M
    zs = (Z0 + n + margin - 1 - np.arange(N)) * STEP_M          # rows along the generator's +y
    x_m, y_m = np.meshgrid(xs, world - zs)
    base = X.tiles_ground(ctx, data, x_m, y_m)
    # the 16 m ecology cells (a cell is water when most of its corners are); the tiles
    # take their canopy and species from them
    dry = ~base["under"]
    c = N // ECO
    cell = lambda a: a.reshape(c, ECO, c, ECO).mean(axis=(1, 3))
    cx, cy = cell(x_m), cell(y_m)
    dens, species, biome, mods = X.forest_points(ctx, data, cx, cy, cell(dry.astype(np.float64)) >= 0.5, float(ECO * STEP_M))
    up = lambda a: np.repeat(np.repeat(a, ECO, 0), ECO, 1)
    canopy, sp = up(dens), up(species)
    t = X.tiles_cover(ctx, data, x_m, y_m, base, canopy, sp)
    inner = slice(margin, margin + n)
    ci = slice(margin // ECO, (margin + n) // ECO)
    flip = lambda a: np.ascontiguousarray(a[::-1])
    raw_h = np.clip(np.rint((t["h"].astype(np.float64) - ORIGIN_M) / UNIT_M), 0, 65535).astype("<u2")
    raw_t = X.tile_raw(t["ground"], t["cover"], t["growth"])
    raw_d = np.clip(np.rint(base["depth"] / SOIL_UNIT_M), 0, 255).astype(np.uint8)
    eco = (biome.astype(np.uint16) & 0x3F) | ((dens.astype(np.uint16) & 0xF) << 6) | ((species.astype(np.uint16) & 0x3F) << 10)
    mod = sum((mods[k].astype(np.uint16) & 0xF) << (4 * i) for i, k in enumerate(F.MODIFIERS))
    still = np.where(t["lake"] > 0, t["lake"], np.where(t["pond"] > 0, t["pond"] + data["_lake_count"], 0))
    still_cell = still.reshape(c, ECO, c, ECO).max(axis=(1, 3))
    return {"h": flip(raw_h[inner, inner]), "t": flip(raw_t[inner, inner]), "d": flip(raw_d[inner, inner]), "eco": flip(eco[ci, ci].astype("<u2")),
            "mod": flip(mod[ci, ci].astype("<u2")), "still": flip(still_cell[ci, ci].astype("<u2")),
            "land": int((t["h"][inner, inner] > 0).sum())}


def _file(job):
    fx, fz = job
    ctx, data = _W["ctx"], _W["data"]
    t0 = time.time()
    H = np.zeros((FILE, FILE), "<u2")
    T = np.zeros((FILE, FILE), "<u2")
    D = np.zeros((FILE, FILE), np.uint8)
    E = np.zeros((FILE // ECO, FILE // ECO), "<u2")
    M = np.zeros_like(E)
    S = np.zeros_like(E)
    land = 0
    for bz in range(0, FILE, BLOCK):
        for bx in range(0, FILE, BLOCK):
            b = block(ctx, data, fx * FILE + bx, fz * FILE + bz)
            H[bz:bz + BLOCK, bx:bx + BLOCK] = b["h"]
            T[bz:bz + BLOCK, bx:bx + BLOCK] = b["t"]
            D[bz:bz + BLOCK, bx:bx + BLOCK] = b["d"]
            e = slice(bz // ECO, (bz + BLOCK) // ECO), slice(bx // ECO, (bx + BLOCK) // ECO)
            E[e], M[e], S[e] = b["eco"], b["mod"], b["still"]
            land += b["land"]
    out = _W["out"]
    H.tofile(Path(out["heights_dir"]) / f"h_{fx:02}_{fz:02}.u16")
    T.tofile(Path(out["tiles_dir"]) / f"t_{fx:02}_{fz:02}.u16")
    D.tofile(Path(out["soil_dir"]) / f"s_{fx:02}_{fz:02}.u8")
    fh, ft, fe = FILE // HEIGHT_STRIDE, FILE // TILE_STRIDE, FILE // ECO
    far_h = _open("far_heights", (CORNERS // HEIGHT_STRIDE,) * 2)
    far_h[fz * fh:(fz + 1) * fh, fx * fh:(fx + 1) * fh] = H[::HEIGHT_STRIDE, ::HEIGHT_STRIDE]
    far_t = _open("far_tiles", (CORNERS // TILE_STRIDE,) * 2)
    far_t[fz * ft:(fz + 1) * ft, fx * ft:(fx + 1) * ft] = T[::TILE_STRIDE, ::TILE_STRIDE] & 0x0FFF    # growth cleared
    for name, a in (("ecology", E), ("modifiers", M), ("still", S)):
        mm = _open(name, (CORNERS // ECO,) * 2)
        mm[fz * fe:(fz + 1) * fe, fx * fe:(fx + 1) * fe] = a
        mm.flush()
    far_h.flush()
    far_t.flush()
    return fx, fz, land, time.time() - t0


# ---------------------------------------------------------------------- tables
def to_z(ctx, y):
    return ctx.world_m - np.asarray(y, np.float64)


def water_tables(ctx, data, package):
    """Lakes and ponds (the still water raster's ids), rivers and waterfalls."""
    wdir = package / "water"
    wdir.mkdir(parents=True, exist_ok=True)
    bodies = []
    for L in data["water_lakes"]:
        bodies.append(dict(id=int(L["id"]), kind=L["kind"], level_m=round(L["level_m"], 2),
                           area_km2=round(L["area_km2"], 4), fetch_km=round(L["fetch_km"], 3),
                           max_depth_m=round(L["max_depth_m"], 1)))
    nl = data["_lake_count"]
    kinds = {1: "pond", 2: "tarn", 3: "oxbow", 4: "spring pool"}
    for k, pd in enumerate(data["coast_ponds"]):
        bodies.append(dict(id=nl + k + 1, kind=kinds.get(int(pd[7]), "pond"), level_m=round(float(pd[5]), 2),
                           x_m=round(float(pd[0]), 1), z_m=round(float(to_z(ctx, pd[1])), 1),
                           a_m=round(float(pd[2]), 1), b_m=round(float(pd[3]), 1), depth_m=round(float(pd[6]), 2)))
    (wdir / "still.json").write_text(json.dumps(dict(
        about="Still water: lakes and ponds. water/still_16m.u16 (4000x4000, rows from the south) holds the id of the "
              "body covering each 16 m cell (0 none); a corner is under it where its height is below level_m. "
              "The sea is everything below 0 m that no body covers.", bodies=bodies), indent=1), encoding="utf-8")
    lines = data["stream_lines"]
    pts, index = [], []
    k = 0
    for line in lines:
        a = np.asarray(line, np.float64)
        q = np.stack([a[:, 0], to_z(ctx, a[:, 1]), a[:, 3], a[:, 2], a[:, 5], a[:, 6], a[:, 4], a[:, 7]], -1)
        pts.append(q.astype("<f4"))
        index.append([k, len(a)])
        k += len(a)
    if pts:
        np.concatenate(pts).tofile(wdir / "rivers.f32")
    falls = [dict(x_m=round(float(x), 1), z_m=round(float(to_z(ctx, y)), 1), drop_m=round(float(dr), 1),
                  width_m=round(float(w), 1)) for x, y, dr, w in data["waterfalls"]]
    (wdir / "rivers.json").write_text(json.dumps(dict(
        about="Channel centrelines, source to mouth. rivers.f32: little-endian float32 points of 8 values: x_m, z_m, "
              "water surface (m), width (m), depth (m), speed (m/s), discharge (m3/s), seasonal (1 = dries in "
              "summer). lines: [first point, point count]. Channels narrower than a 2 m tile are carved as one tile "
              "of water.", point_fields=["x_m", "z_m", "surface_m", "width_m", "depth_m", "speed_mps", "discharge_m3s",
                                         "seasonal"], lines=index, waterfalls=falls)), encoding="utf-8")
    return len(bodies), len(lines)


def spawn_corner(ctx, data):
    """A dry corner on open, gentle, fertile ground near the middle of the largest landmass."""
    from scipy import ndimage
    land = data["land"] & (data["water_lake_id"] == 0)
    lab, n = ndimage.label(land)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else land
    cy, cx = ndimage.center_of_mass(big)
    ok = big & (data["biome"] == B.MEADOW) & (data["slope"] < 0.05) & (data["fertility"] >= 0.5) \
        & (data["coast_dist_km"] > 1.0) & (data["channel_class"] == 0)
    if not ok.any():
        ok = big
    ys, xs = np.nonzero(ok)
    i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    x_m, y_m = (xs[i] + 0.5) * ctx.cell_m, (ys[i] + 0.5) * ctx.cell_m
    # the nearest dry, gentle corner in a small window around it
    X0, Z0 = int(x_m / STEP_M) - 32, int(to_z(ctx, y_m) / STEP_M) - 32
    b = X.tiles(ctx, data, *np.meshgrid((X0 + np.arange(64)) * STEP_M, ctx.world_m - (Z0 + 63 - np.arange(64)) * STEP_M))
    h = b["h"][::-1]
    dry = (h > 1.0) & ~(b["water"] | b["sea"])[::-1]
    gy, gx = np.gradient(h.astype(np.float64), STEP_M)
    dry &= np.hypot(gx, gy) < 0.15
    zz, xx = np.nonzero(dry) if dry.any() else (np.array([32]), np.array([32]))
    j = np.argmin((zz - 32) ** 2 + (xx - 32) ** 2)
    return int(np.clip(X0 + xx[j], 0, CORNERS - 1)), int(np.clip(Z0 + zz[j], 0, CORNERS - 1))


# ---------------------------------------------------------------------- main
def run(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    full = Path(job["full_dir"])
    package = Path(job["package_dir"])
    full.mkdir(parents=True, exist_ok=True)
    st = full / "export_status.json"
    t_start = time.time()
    status(st, state="running", step="pipeline", done=0, total=FILES * FILES, map_id=job["map_id"])
    pipe = Pipeline(STAGES)
    ctx, data, _ = pipe.compute(job["seed"], job["res"], job["params"], "export", job["scale"], job["vscale"])
    data["_lake_count"] = len(data["water_lakes"])
    if data["_lake_count"] + len(data["coast_ponds"]) > 65535:
        raise RuntimeError("more lakes and ponds than the still-water raster's 16-bit ids hold")
    work = full / "_work"
    save_data(ctx, data, work)
    (full / "heights").mkdir(exist_ok=True)
    (full / "tiles").mkdir(exist_ok=True)
    (full / "soil").mkdir(exist_ok=True)
    for d in ("far", "ecology", "water"):
        (package / d).mkdir(parents=True, exist_ok=True)
    out = {"heights_dir": str(full / "heights"), "tiles_dir": str(full / "tiles"), "soil_dir": str(full / "soil"),
           "far_heights": str(package / "far" / "heights_8m.u16"), "far_tiles": str(package / "far" / "tiles_16m.u16"),
           "ecology": str(package / "ecology" / "ecology_16m.u16"), "modifiers": str(package / "ecology" / "modifiers_16m.u16"),
           "still": str(package / "water" / "still_16m.u16")}
    for name, side in (("far_heights", CORNERS // HEIGHT_STRIDE), ("far_tiles", CORNERS // TILE_STRIDE),
                       ("ecology", CORNERS // ECO), ("modifiers", CORNERS // ECO), ("still", CORNERS // ECO)):
        np.memmap(out[name], dtype="<u2", mode="w+", shape=(side, side)).flush()

    status(st, state="running", step="spawn and water tables", done=0, total=FILES * FILES, map_id=job["map_id"])
    spawn = spawn_corner(ctx, data)
    n_bodies, n_lines = water_tables(ctx, data, package)

    files = [(fx, fz) for fz in range(FILES) for fx in range(FILES)]
    if job.get("files"):                         # a subset, for tests
        files = [tuple(f) for f in job["files"]]
    written, land = [], 0
    t1 = time.time()
    with mp.get_context("spawn").Pool(int(job.get("workers", 8)), _init, (str(work), out)) as pool:
        for fx, fz, lc, dt in pool.imap_unordered(_file, files):
            written.append(f"{fx:02}_{fz:02}")
            land += lc
            left = (time.time() - t1) / len(written) * (len(files) - len(written))
            status(st, state="running", step="tiles", done=len(written), total=len(files), map_id=job["map_id"],
                   eta_s=round(left))
    written.sort(key=lambda s: (s[3:], s[:2]))

    params_json = json.dumps(job["params"], sort_keys=True)
    manifest = {
        "mapId": job["map_id"], "generatorVersion": GENERATOR_VERSION, "format_version": 1,
        "cornersX": CORNERS, "cornersZ": CORNERS, "corners_per_side": CORNERS, "file_size": FILE, "files_per_side": FILES,
        "height_unit_m": UNIT_M, "height_origin_m": ORIGIN_M, "seaLevelRaw": SEA_LEVEL_RAW, "sea_level_raw": SEA_LEVEL_RAW,
        "heightStride": HEIGHT_STRIDE, "tileStride": TILE_STRIDE, "tiles_have_growth": True,
        "far_tiles_growth_cleared": True,
        "axes": "corner (x, z) = world (2x, 2z) m, +X east, +Z north, row 0 = z 0 = south edge, little-endian uint16",
        "floorDepthM": float(data["relief_params"]["floor_m"]),
        "ecologyStride": ECO, "ecologyVersion": ECOLOGY_VERSION,
        "soilDepth": {"files": "soil/s_{fx:02}_{fz:02}.u8", "unit_m": SOIL_UNIT_M, "max_m": 255 * SOIL_UNIT_M,
                      "about": "soil over the rock at each corner (same grid, files and rows as the heights); rock "
                               "height = corner height - depth; 0 at every corner of a rock tile, at least 0.1 m "
                               "elsewhere on land"},
        "ecology": {"file": "ecology/ecology_16m.u16", "bits": "biome 0-5, tree density 6-9, species 10-15",
                    "biomes": [n for n, _ in B.BIOMES], "species": [n for n, _, _ in F.SPECIES],
                    "modifiers_file": "ecology/modifiers_16m.u16", "modifier_bits": "flowers 0-3, rockiness 4-7, "
                    "undergrowth 8-11, dead wood 12-15"},
        "water": {"still_raster": "water/still_16m.u16", "still_table": "water/still.json", "rivers": "water/rivers.json",
                  "river_points": "water/rivers.f32", "bodies": n_bodies, "rivers_count": n_lines},
        "files_written": written, "complete": len(written) == len(files),
        "seed": job["seed"], "worldScale": job["scale"], "heightsFollowScale": job["vscale"] < 1.0,
        "macro_resolution": job["res"], "params": job["params"],
        "params_sha256": hashlib.sha256(params_json.encode()).hexdigest(),
        "spawnCornerX": spawn[0], "spawnCornerZ": spawn[1],
    }
    text = json.dumps(manifest, indent=2)
    (full / "manifest.json").write_text(text, encoding="utf-8")
    (package / "manifest.json").write_text(text, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    status(st, state="done", step="done", done=len(files), total=len(files), map_id=job["map_id"],
           minutes=round((time.time() - t_start) / 60, 1), land_km2=round(land * STEP_M * STEP_M / 1e6),
           spawn=list(spawn), full_dir=str(full), package_dir=str(package))


if __name__ == "__main__":
    job_file = sys.argv[1]
    try:
        run(job_file)
    except Exception as e:
        job = json.loads(Path(job_file).read_text(encoding="utf-8"))
        status(Path(job["full_dir"]) / "export_status.json", state="failed", error=f"{type(e).__name__}: {e}",
               trace=traceback.format_exc()[-3000:], map_id=job.get("map_id"))
        raise
