"""Local server of the world generator tuning tool (WORLDGEN.md section 4).

    .venv\\Scripts\\python server.py [--port 8765] [--open]

Serves the page in ui/ and a small JSON API:
    GET  /api/stages              stage list with parameters and views, defaults
    POST /api/render              {seed, res, scale, height_follow, params, stage, view, height} -> image (+ heights for 3D)
    POST /api/detail              {seed, res, params, stage, cx, cy, size_km, height} -> high-resolution window
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

from worldgen.core import colormaps
from worldgen.core.pipeline import Pipeline
from worldgen.stages import STAGES

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
EXPORT = {"proc": None, "full": None}
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


def export_start(body):
    """Start the 2 m export (worldgen/export_job.py) as its own process."""
    if EXPORT["proc"] is not None and EXPORT["proc"].poll() is None:
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
    log = open(full / "export.log", "w", encoding="utf-8")
    # its own process group, detached from this console: stopping or restarting the tool
    # (Ctrl+C, closing the window) must not kill an export half-way
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    EXPORT["proc"] = subprocess.Popen([sys.executable, "-m", "worldgen.export_job", str(full / "export_job.json")],
                                      cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                      start_new_session=os.name != "nt")
    EXPORT["full"] = full
    return {"started": map_id, "full_dir": str(full), "package_dir": str(package)}


def export_status():
    full = EXPORT["full"]
    if full is None:
        # after a restart of the tool: the most recent export on disk (it may still be running)
        found = sorted(FULL_ROOT.glob("*/export_status.json"), key=lambda p: p.stat().st_mtime)
        if not found:
            return {"state": "none"}
        try:
            return json.loads(found[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"state": "none"}
    try:
        st = json.loads((full / "export_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {"state": "running", "step": "starting"}
    running = EXPORT["proc"] is not None and EXPORT["proc"].poll() is None
    if st.get("state") == "running" and not running:
        log = (full / "export.log").read_text(encoding="utf-8", errors="replace")[-1500:]
        st = dict(st, state="failed", error="the export process stopped", trace=log)
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
