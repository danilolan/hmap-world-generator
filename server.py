"""Local server of the world generator tuning tool (WORLDGEN.md section 4).

    .venv\\Scripts\\python server.py [--port 8765] [--open]

Serves the page in ui/ and a small JSON API:
    GET  /api/stages              stage list with parameters and views, defaults
    POST /api/render              {seed, res, scale, height_follow, params, stage, view, height} -> image (+ heights for 3D)
    POST /api/detail              {seed, res, params, stage, cx, cy, size_km, height} -> high-resolution window
    POST /api/tiles               {seed, res, scale, height_follow, params, cx, cy, corners} -> the exported 2 m data of a
                                  small window (heights, tiles, soil, ecology, water), for the page's 3D tile view
    POST /api/gallery             {seeds, res, params, stage, view} -> thumbnails
    POST /api/export              {seed, res, scale, height_follow, params, map_id, overwrite} -> starts the 2 m export
    GET  /api/export              the running or last export's progress
    GET  /api/presets             preset names
    GET  /api/presets/<name>      one preset
    POST /api/presets             {name, data} -> save
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy import ndimage

from worldgen import export_job as J
from worldgen.core import colormaps
from worldgen.core.pipeline import Pipeline
from worldgen.stages import STAGES
from worldgen.stages import biomes as B
from worldgen.stages import export as X
from worldgen.stages import forests as F

HERE = Path(__file__).parent
UI = HERE / "ui"
PRESETS = HERE / "presets"
RESOLUTIONS = [256, 512, 1024, 2048]
# the game repository (its client package receives the exported map): MEGASURVIVAL_GAME, or
# the MegaSurvival folder beside this repository
GAME = Path(os.environ.get("MEGASURVIVAL_GAME", HERE.parent / "MegaSurvival"))
FULL_ROOT = GAME.parent / "MegaSurvivalWorld"                       # full-resolution files, outside both repositories
PACKAGE_ROOT = GAME / "Assets" / "StreamingAssets" / "World"       # the client package
PROTECTED_MAPS = {"worldgen-dev"}                                  # the map in the game today (WORLDGEN.md section 6)
EXPORT = {"full": None, "launched": 0.0}
PIPELINE = Pipeline(STAGES)
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png"}


def stage_list():
    return [dict(id=s.id, title=s.title, description=s.description, params=[p.to_json() for p in s.params],
                 views=[dict(id=k, label=v) for k, v in s.views.items()], has_height=s.height_output is not None,
                 has_detail=s.has_detail)
            for s in STAGES]


def scales(body):
    """World scale from the top bar: the map shows a world `scale` times larger, compressed. Heights
    shrink with it (vscale = 1/scale) unless the page asks to keep them."""
    scale = min(max(float(body.get("scale", 1.0)), 1.0), 8.0)
    return dict(scale=scale, vscale=1.0 / scale if body.get("height_follow", True) else 1.0)


def render(body, thumbnail_res=None):
    stage = STAGES[PIPELINE.index(body["stage"])]
    res = thumbnail_res or int(body.get("res", 512))
    ctx, data, timings = PIPELINE.compute(int(body.get("seed", 1)), res, body.get("params", {}), stage.id, **scales(body))
    view = body.get("view") or next(iter(stage.views))
    with colormaps.recording() as ramps:
        img = stage.render(view, ctx, data)
    out = dict(image=colormaps.data_url(img), stats=stage.stats(ctx, data), legend=legend(stage, view, ctx, data, ramps),
               timings=[dict(stage=s, ms=round(t * 1000), cached=c) for s, t, c in timings])
    if body.get("height") and stage.height_output:
        h = data[stage.height_output].astype(np.float32)
        m = 256
        if h.shape[0] != m:
            h = ndimage.zoom(h, m / h.shape[0], order=1).astype(np.float32)
        out["height"] = dict(n=m, cell_m=ctx.world_m / m, data=base64.b64encode(h.tobytes()).decode())
    return out


def legend(stage, view, ctx, data, ramps):
    """The view's legend: the stage's own list of colours, or the colour bar of the first
    ramp the view used."""
    items = stage.legend(view, ctx, data)
    if items:
        out = dict(kind="items", title=stage.views.get(view, view), items=items)
        labels = stage.legend_labels(view, ctx, data)
        if labels is not None:
            out["labels"] = label_url(labels, ctx.res)
        return out
    if ramps:
        return dict(ramps[0], title=stage.views.get(view, view), unit=stage.units.get(view, ""))
    return None


def label_url(labels, n):
    """Legend item per pixel as a grey PNG (item index + 1, 0 = none), at the image's size."""
    labels = np.asarray(labels)
    if labels.shape[0] != n:
        labels = ndimage.zoom(labels, n / labels.shape[0], order=0)
    return colormaps.data_url(np.clip(labels + 1, 0, 255).astype(np.uint8))


def detail_labels(items, h, water, overlays):
    """Legend items of a detail window: its overlays matched to the legend by colour, as
    they were drawn (later ones on top), and the water."""
    lab = np.full(h.shape, -1, np.int16)
    colors = {tuple(int(c) for c in it["color"]): i for i, it in enumerate(items)}
    for mask, rgb in overlays:
        i = colors.get(tuple(int(round(c)) for c in rgb))
        if i is not None:
            lab[mask] = i
    wi = next((i for i, it in enumerate(items) if it["label"] == "water"), None)
    if water is not None and wi is not None:
        lab[water & (h > 0)] = wi
    return lab


def detail(body):
    """High-resolution window around a point: the stage's macro result plus its micro relief."""
    stage = STAGES[PIPELINE.index(body["stage"])]
    ctx, data, timings = PIPELINE.compute(int(body.get("seed", 1)), int(body.get("res", 512)), body.get("params", {}), stage.id,
                                        **scales(body))
    p = stage.coerce((body.get("params") or {}).get(stage.id))
    size = float(body.get("size_km", 2.0)) * 1000.0
    m = 512
    g = (np.arange(m) + 0.5) / m * size - size / 2
    cx, cy = float(body.get("cx", 0.5)) * ctx.world_m, float(body.get("cy", 0.5)) * ctx.world_m
    X, Y = np.meshgrid(cx + g, cy + g)
    res = stage.detail(ctx, data, p, X, Y)
    detail_legend = stage.legend("detail", ctx, data) or stage.legend(body.get("view") or next(iter(stage.views)), ctx, data)
    # a stage's detail returns heights, or (heights, water mask[, [(mask, rgb) overlays]])
    h, water, overlays = (tuple(res) + ([],))[:3] if isinstance(res, tuple) else (res, None, [])
    img = colormaps.terrain(h, size / m)
    for mask, rgb in overlays:
        img[mask] = (0.4 * img[mask] + 0.6 * np.array(rgb)).astype(np.uint8)
    if water is not None:
        img[water & (h > 0)] = (55, 115, 200)
    out = dict(image=colormaps.data_url(img),
               stats={"window": f"{size / 1000:g} km at ({cx / 1000:.1f}, {cy / 1000:.1f}) km",
                      "cell (m)": round(size / m, 1),
                      "sampled from": f"the {ctx.res}² grid ({ctx.cell_m:.0f} m): raise Preview for sharper detail",
                      "lowest (m)": int(h.min()), "highest (m)": int(h.max()),
                      "relief in window (m)": int(h.max() - h.min())},
               legend=dict(kind="items", title="Detail", items=detail_legend,
                           labels=label_url(detail_labels(detail_legend, h, water, overlays), m))
               if detail_legend else None,
               timings=[dict(stage=s, ms=round(t * 1000), cached=c) for s, t, c in timings])
    if body.get("height"):
        hs = ndimage.zoom(h, 256 / m, order=1).astype(np.float32)
        out["height"] = dict(n=256, cell_m=size / 256, data=base64.b64encode(hs.tobytes()).decode())
    return out


TILE_SIZES = (64, 128)            # corners per side of the 3D tile window: 128 m or 256 m
TILE_BORDER = 4                   # corners around it: an edge tile's surface reads one corner beyond it
TILES = {"key": None}             # the pipeline result the last tile window used (its memo() values stay warm)
COVER_NAMES = {X.NO_COVER: "none", X.GRASS: "grass", X.DRY_GRASS: "dry grass"}


def tiles_block(ctx, data, cx, cy, corners):
    """The 2 m export of the window of `corners` corners (plus TILE_BORDER each side) around
    the point (cx, cy), fractions of the map with row 0 = north: export_job.block() itself,
    at an origin on the 16 m ecology grid, so every value is the one Export writes there.
    Returns the block and its corner origin (x0, z0), rows from the south."""
    n = corners + 2 * TILE_BORDER
    snap = lambda c: int(np.clip(np.floor((c - n / 2) / J.ECO) * J.ECO, 0, J.CORNERS - n))
    x0 = snap(cx * ctx.world_m / J.STEP_M)
    z0 = snap((1.0 - cy) * ctx.world_m / J.STEP_M)
    data.setdefault("_lake_count", len(data["water_lakes"]))
    return J.block(ctx, data, x0, z0, n=n), x0, z0


def still_levels(data, ids):
    """Water level (m) of the lakes and ponds `ids` of the still-water raster (export_job.water_tables)."""
    nl = data["_lake_count"]
    lakes = {int(L["id"]): float(L["level_m"]) for L in data["water_lakes"]}
    ponds = data["coast_ponds"]
    return {int(i): lakes.get(int(i)) if i <= nl else float(ponds[int(i) - nl - 1][5]) for i in ids if i > 0}


def rivers_in(ctx, data, x0, z0, n):
    """River centreline pieces crossing the window, as the export writes them (rivers.f32):
    lists of [x_m, z_m, water surface m, width m], world metres."""
    lo_x, hi_x = x0 * J.STEP_M, (x0 + n) * J.STEP_M
    lo_z, hi_z = z0 * J.STEP_M, (z0 + n) * J.STEP_M
    out = []
    for line in data["stream_lines"]:
        a = np.asarray(line, np.float64)
        z = J.to_z(ctx, a[:, 1])
        pad = a[:, 2] + 4.0
        inside = (a[:, 0] + pad > lo_x) & (a[:, 0] - pad < hi_x) & (z + pad > lo_z) & (z - pad < hi_z)
        if not inside.any():
            continue
        k = np.flatnonzero(inside)
        # runs of consecutive points, one point beyond each end so the ribbon reaches the edge
        for run in np.split(k, np.flatnonzero(np.diff(k) > 1) + 1):
            s = slice(max(run[0] - 1, 0), min(run[-1] + 2, len(a)))
            out.append(np.round(np.stack([a[s, 0], z[s], a[s, 3], a[s, 2]], -1), 2).tolist())
    return out


def tiles_window(body):
    """The 3D tile view's window: what Export writes around a point, with the seed, grid,
    scale and parameters the page sends (its Grid choice is the export's)."""
    world, res, seed, params = scales(body), int(body.get("res", 2048)), int(body.get("seed", 1)), body.get("params", {})
    key = json.dumps([seed, res, world, params], sort_keys=True)
    if TILES["key"] != key:
        ctx, data, timings = PIPELINE.compute(seed, res, params, "export", **world)
        TILES.update(key=key, ctx=ctx, data=data)
    else:
        timings = []
    ctx, data = TILES["ctx"], TILES["data"]
    corners = int(body.get("corners", 64))
    corners = corners if corners in TILE_SIZES else TILE_SIZES[0]
    t0 = time.perf_counter()
    b, x0, z0 = tiles_block(ctx, data, float(body.get("cx", 0.5)), float(body.get("cy", 0.5)), corners)
    n = b["h"].shape[0]
    t = b["t"]
    ground, cover, growth = t & 0xFF, (t >> 8) & 0x0F, t >> 12
    rgb = X.tile_rgb({"ground": ground, "cover": cover, "growth": growth}).astype(np.uint8)
    h = b["h"].astype(np.float64) * J.UNIT_M + J.ORIGIN_M
    inner = (slice(TILE_BORDER, n - TILE_BORDER),) * 2
    land = h[inner] > 0
    shares = lambda a, names: ", ".join(f"{names.get(int(k), k)} {c / a.size * 100:.0f}%"
                                        for k, c in zip(*np.unique(a, return_counts=True)))
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
    stats = {"window": f"{corners * J.STEP_M:g} m at ({x0 * J.STEP_M / 1000:.2f}, {z0 * J.STEP_M / 1000:.2f}) km "
                       f"(corner {x0 + TILE_BORDER}, {z0 + TILE_BORDER})",
             "grid": f"{res}²", "lowest (m)": round(float(h[inner].min()), 1), "highest (m)": round(float(h[inner].max()), 1),
             "ground": shares(ground[inner], X.GROUND_NAMES), "cover": shares(cover[inner], COVER_NAMES),
             "soil depth on land (m)": f"{b['d'][inner][land].min() * J.SOIL_UNIT_M:.1f}–"
                                       f"{b['d'][inner][land].max() * J.SOIL_UNIT_M:.1f}" if land.any() else "no land"}
    return dict(x0=x0, z0=z0, n=n, border=TILE_BORDER, eco=J.ECO, step_m=J.STEP_M, world_m=ctx.world_m, unit_m=J.UNIT_M, origin_m=J.ORIGIN_M,
                soil_unit_m=J.SOIL_UNIT_M, h=b64(b["h"]), t=b64(t), d=b64(b["d"]), rgb=b64(rgb), ecology=b64(b["eco"]),
                modifiers=b64(b["mod"]), still=b64(b["still"]), still_levels=still_levels(data, np.unique(b["still"])),
                rivers=rivers_in(ctx, data, x0, z0, n),
                ground_names={int(k): v for k, v in X.GROUND_NAMES.items()},
                cover_names={int(k): v for k, v in COVER_NAMES.items()},
                biomes=[dict(name=nm, color=list(c)) for nm, c in B.BIOMES],
                species=[dict(name=nm, color=list(c), kind=int(k)) for nm, c, k in F.SPECIES],
                modifier_names=list(F.MODIFIERS), stats=stats,
                legend=dict(kind="items", title="Tile materials", items=X.Export().legend("tiles", ctx, data)),
                block_ms=round((time.perf_counter() - t0) * 1000),
                timings=[dict(stage=s, ms=round(tm * 1000), cached=c) for s, tm, c in timings])


def launch_detached(cmd, log_path):
    """Start a process that outlives this tool, its output going to `log_path`. On Windows
    it is created by the WMI service (Win32_Process.Create), so it belongs to no console,
    process group or job object of whoever started the tool: a job kills every process in
    it when it closes, and a child that only detached from the console still died with the
    tool's session. Stopping or restarting the tool must not kill an export half-way."""
    if os.name != "nt":
        subprocess.Popen(cmd, cwd=str(HERE), stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        return
    line = subprocess.list2cmdline(cmd) + f' > "{log_path}" 2>&1'
    ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
          f"@{{CommandLine='cmd /c \"{line}\"'; CurrentDirectory='{HERE}'}}; $r.ReturnValue")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    if out.stdout.strip() != "0":
        raise RuntimeError(f"could not start the export process: {out.stdout.strip()} {out.stderr.strip()}")


def pid_alive(pid):
    """Whether a process is still running (Windows: its exit code is still STILL_ACTIVE)."""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))           # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259


def export_start(body):
    """Start the 2 m export (worldgen/export_job.py) as its own process."""
    if export_status().get("state") == "running":
        return {"error": "an export is already running"}
    map_id = str(body.get("map_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", map_id):
        return {"error": "map id: letters, digits, - and _ only"}
    if map_id in PROTECTED_MAPS:
        return {"error": f"{map_id} is the map the game uses today: export to a new id"}
    full, package = FULL_ROOT / map_id, PACKAGE_ROOT / map_id
    if (package.exists() or (full / "manifest.json").exists()) and not body.get("overwrite"):
        return {"exists": True, "full_dir": str(full), "package_dir": str(package)}
    full.mkdir(parents=True, exist_ok=True)
    job = dict(seed=int(body.get("seed", 1)), res=max(1024, int(body.get("res", 2048))), params=body.get("params", {}),
               map_id=map_id, full_dir=str(full), package_dir=str(package),
               workers=max(1, min((os.cpu_count() or 4) - 2, 16)), **scales(body))
    (full / "export_job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    (full / "export_status.json").unlink(missing_ok=True)
    launch_detached([sys.executable, "-m", "worldgen.export_job", str(full / "export_job.json")], full / "export.log")
    EXPORT["full"], EXPORT["launched"] = full, time.time()
    return {"started": map_id, "full_dir": str(full), "package_dir": str(package)}


STALLED_S = 30 * 60      # no progress report for this long: the export is taken as dead


def export_status():
    """The running or last export's progress. A process that died, or one silent for
    STALLED_S (the pipeline step reports nothing for up to ~10 min at 2048²), is reported as
    failed, with the end of its log."""
    full = EXPORT["full"]
    known = full is not None
    if not known:
        # after a restart of the tool: an export on disk still running (its process alive),
        # else the most recent one
        found = sorted(FULL_ROOT.glob("*/export_status.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            return {"state": "none"}
        full = found[-1].parent
        for f in reversed(found):
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if s.get("state") == "running" and pid_alive(s.get("pid")):
                full = f.parent
                break
    tail = lambda: (full / "export.log").read_text(encoding="utf-8", errors="replace")[-3000:] \
        if (full / "export.log").exists() else ""
    try:
        st = json.loads((full / "export_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        if known and time.time() - EXPORT["launched"] > 120:
            st = {"state": "failed", "step": "starting", "error": "the export process never started", "trace": tail()}
        else:
            st = {"state": "running", "step": "starting", "started": EXPORT["launched"] or time.time(), "time": time.time()}
    if st.get("state") == "running":
        # every progress report carries the export's process id: gone means it died
        if st.get("pid") and not pid_alive(st["pid"]):
            st = dict(st, state="failed", error="the export process stopped unexpectedly (see the log below)",
                      trace=tail() or "(the log is empty: the process was killed from outside)")
        elif time.time() - st.get("time", time.time()) > STALLED_S:
            st = dict(st, state="failed", error=f"no progress for {STALLED_S // 60} min: the export process has "
                                                 "probably stopped", trace=tail())
    st["now"] = time.time()
    st.setdefault("full_dir", str(full))
    return st


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, (UI / "index.html").read_bytes(), "text/html")
            if path.startswith("/ui/"):
                f = (UI / path[4:]).resolve()
                if UI.resolve() in f.parents and f.exists():
                    return self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
                return self._send(404, {"error": "not found"})
            if path == "/api/stages":
                return self._send(200, dict(stages=stage_list(), defaults=PIPELINE.defaults(),
                                            resolutions=RESOLUTIONS, world_m=PIPELINE.world_m))
            if path == "/api/presets":
                return self._send(200, sorted(p.stem for p in PRESETS.glob("*.json")))
            if path == "/api/export":
                return self._send(200, export_status())
            m = re.fullmatch(r"/api/presets/([\w\- ]+)", path)
            if m:
                f = PRESETS / f"{m.group(1)}.json"
                return self._send(200, json.loads(f.read_text())) if f.exists() else self._send(404, {"error": "no preset"})
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/render":
                t0 = time.perf_counter()
                out = render(body)
                out["ms"] = round((time.perf_counter() - t0) * 1000)
                return self._send(200, out)
            if self.path == "/api/detail":
                t0 = time.perf_counter()
                out = detail(body)
                out["ms"] = round((time.perf_counter() - t0) * 1000)
                return self._send(200, out)
            if self.path == "/api/tiles":
                t0 = time.perf_counter()
                out = tiles_window(body)
                out["ms"] = round((time.perf_counter() - t0) * 1000)
                return self._send(200, out)
            if self.path == "/api/export":
                return self._send(200, export_start(body))
            if self.path == "/api/gallery":
                res = int(body.get("res", 128))
                thumbs = [dict(seed=s, image=render(dict(body, seed=s), res)["image"]) for s in body.get("seeds", [])]
                return self._send(200, thumbs)
            if self.path == "/api/presets":
                name = re.sub(r"[^\w\- ]", "", str(body.get("name", ""))).strip()
                if not name:
                    return self._send(400, {"error": "empty name"})
                PRESETS.mkdir(exist_ok=True)
                (PRESETS / f"{name}.json").write_text(json.dumps(body.get("data", {}), indent=2))
                return self._send(200, {"saved": name})
            self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()
    url = f"http://localhost:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"World generator tool: {url}  (Ctrl+C to stop)", flush=True)
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    server.serve_forever()
