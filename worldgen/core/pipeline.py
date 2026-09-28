"""Runs the stages in order with a cache.

Each stage's outputs are cached under a key chained from the seed, the
resolution, and the parameters of that stage and every stage before it. Moving
a slider of stage k therefore recomputes only stages k and later.
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict

from .stage import Context


class Pipeline:
    def __init__(self, stages, world_m=64000.0, cache_entries=96):
        self.stages = list(stages)
        self.world_m = world_m
        self._cache = OrderedDict()
        self._cache_entries = cache_entries
        self._lock = threading.Lock()

    def index(self, stage_id: str) -> int:
        for i, s in enumerate(self.stages):
            if s.id == stage_id:
                return i
        raise KeyError(stage_id)

    def defaults(self) -> dict:
        return {s.id: s.defaults() for s in self.stages}

    def compute(self, seed: int, res: int, params: dict, upto: str, scale: float = 1.0, vscale: float = 1.0):
        """Run stages up to and including `upto`. Returns (ctx, data, timings) where
        data merges every computed stage's outputs."""
        ctx = Context(int(seed), int(res), self.world_m, float(scale), float(vscale))
        key = hashlib.sha1(json.dumps([ctx.seed, ctx.res, ctx.world_m, ctx.scale, ctx.vscale]).encode()).hexdigest()
        data, timings = {}, []
        last = self.index(upto)
        with self._lock:
            for stage in self.stages[:last + 1]:
                p = stage.coerce((params or {}).get(stage.id))
                key = hashlib.sha1((key + stage.id + json.dumps(p, sort_keys=True)).encode()).hexdigest()
                if key in self._cache:
                    self._cache.move_to_end(key)
                    out = self._cache[key]
                    timings.append((stage.id, 0.0, True))
                else:
                    t0 = time.perf_counter()
                    out = stage.run(ctx, dict(data), p)
                    timings.append((stage.id, time.perf_counter() - t0, False))
                    self._cache[key] = out
                    while len(self._cache) > self._cache_entries:
                        self._cache.popitem(last=False)
                data.update(out)
        return ctx, data, timings
