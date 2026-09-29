"""Checks that the tool's 3D tile window (server.tiles_block) returns exactly what an export wrote:

    .venv\\Scripts\\python -m tests.check_tiles_window <full_dir> [windows]

<full_dir> is an export's full-resolution folder (MegaSurvivalWorld/<map id>) whose _work folder
still holds the pipeline data (an export running, or one kept for the check). Windows are picked
at random inside the files already written, some across a file boundary, in both sizes, and every
layer is compared: heights, tiles, soil depth, ecology, modifiers, still water.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402
from worldgen import export_job as J  # noqa: E402


def written(full):
    return {tuple(int(v) for v in f.stem[2:].split("_")) for f in (full / "heights").glob("h_*.u16")
            if f.stat().st_size == J.FILE * J.FILE * 2 and (full / "soil" / f"s_{f.stem[2:]}.u8").exists()}


def main(full, windows=12):
    full = Path(full)
    job = json.loads((full / "export_job.json").read_text(encoding="utf-8"))
    package = Path(job["package_dir"])
    done = written(full)
    if not done:
        raise SystemExit("no complete files yet")
    ctx, data = J.load_data(full / "_work")
    side = J.CORNERS // J.ECO
    eco = {k: np.memmap(package / p, dtype="<u2", mode="r", shape=(side, side))
           for k, p in (("eco", "ecology/ecology_16m.u16"), ("mod", "ecology/modifiers_16m.u16"), ("still", "water/still_16m.u16"))}
    rnd = random.Random(7)
    # pairs of written files side by side: windows across their boundary
    pairs = [(a, (a[0] + 1, a[1])) for a in done if (a[0] + 1, a[1]) in done]
    checked = 0
    for w in range(windows):
        corners = server.TILE_SIZES[w % len(server.TILE_SIZES)]
        n = corners + 2 * server.TILE_BORDER
        if pairs and w % 3 == 0:
            (fx, fz), _ = rnd.choice(pairs)
            xc = (fx + 1) * J.FILE + rnd.randint(-n // 4, n // 4)
            fx_span = (fx, fx + 1)
        else:
            fx, fz = rnd.choice(sorted(done))
            xc = fx * J.FILE + rnd.randint(n, J.FILE - n)
            fx_span = (fx, fx)
        zc = fz * J.FILE + rnd.randint(n, J.FILE - n)
        cx, cy = xc * J.STEP_M / ctx.world_m, 1.0 - zc * J.STEP_M / ctx.world_m
        b, x0, z0 = server.tiles_block(ctx, data, cx, cy, corners)
        assert x0 % J.ECO == 0 and z0 % J.ECO == 0
        # the export's files over the same window
        H = np.concatenate([np.fromfile(full / "heights" / f"h_{f:02}_{fz:02}.u16", "<u2").reshape(J.FILE, J.FILE)
                            for f in range(fx_span[0], fx_span[1] + 1)], 1)
        T = np.concatenate([np.fromfile(full / "tiles" / f"t_{f:02}_{fz:02}.u16", "<u2").reshape(J.FILE, J.FILE)
                            for f in range(fx_span[0], fx_span[1] + 1)], 1)
        D = np.concatenate([np.fromfile(full / "soil" / f"s_{f:02}_{fz:02}.u8", np.uint8).reshape(J.FILE, J.FILE)
                            for f in range(fx_span[0], fx_span[1] + 1)], 1)
        lx, lz = x0 - fx_span[0] * J.FILE, z0 - fz * J.FILE
        if lx < 0 or lz < 0 or lz + n > J.FILE or lx + n > H.shape[1]:
            continue                                   # snapped outside the files read: pick again next round
        win = (slice(lz, lz + n), slice(lx, lx + n))
        cells = (slice(z0 // J.ECO, (z0 + n) // J.ECO), slice(x0 // J.ECO, (x0 + n) // J.ECO))
        for name, got, want in (("heights", b["h"], H[win]), ("tiles", b["t"], T[win]), ("soil", b["d"], D[win]),
                                ("ecology", b["eco"], eco["eco"][cells]), ("modifiers", b["mod"], eco["mod"][cells]),
                                ("still", b["still"], eco["still"][cells])):
            diff = int((np.asarray(got) != np.asarray(want)).sum())
            assert diff == 0, f"window {w} at corner ({x0}, {z0}): {name} differs at {diff} of {np.asarray(want).size}"
        checked += 1
        print(f"window {w}: {corners} corners at ({x0}, {z0}){' across a file boundary' if fx_span[0] != fx_span[1] else ''}: "
              "identical", flush=True)
    print(f"{checked} windows identical to the export")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
