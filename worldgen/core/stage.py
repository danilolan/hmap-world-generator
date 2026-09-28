"""Stage base class and the run context.

A stage reads the outputs of the stages before it, produces named arrays, and
renders views of them. Everything it needs to be tuned (sliders) and shown
(views) is declared on the class, so the UI builds itself.
"""
import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Context:
    seed: int
    res: int                 # grid cells per side
    world_m: float           # world side in metres
    scale: float = 1.0       # world scale: the map shows a world this many times larger, compressed
    vscale: float = 1.0      # height multiplier for macro relief (1/scale when heights follow the scale)

    def km(self, value):
        """A feature length set in the world's own units, as it lands on the map."""
        return value / self.scale

    def height(self, value):
        """A macro height set in the world's own units, as it lands on the map."""
        return value * self.vscale

    @property
    def cell_m(self) -> float:
        return self.world_m / self.res

    def stage_seed(self, stage_id: str) -> int:
        """Per-stage sub-seed derived from the master seed: changing one stage's
        knobs never reshuffles another stage's randomness."""
        digest = hashlib.sha256(f"{self.seed}:{stage_id}".encode()).digest()
        return int.from_bytes(digest[:4], "little")


class Stage:
    id = "stage"
    title = "Stage"
    description = ""
    params = []              # list[Param]
    views = {}               # view id -> label
    height_output = None     # output key used by the 3D view, if any
    has_detail = False       # True if detail() can render a small window at high resolution

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def coerce(self, values: dict) -> dict:
        values = values or {}
        return {p.key: p.coerce(values.get(p.key)) for p in self.params}

    def run(self, ctx: Context, inputs: dict, p: dict) -> dict:
        """Return a dict of named numpy arrays. `inputs` holds every output of the
        stages before this one."""
        raise NotImplementedError

    def render(self, view: str, ctx: Context, data: dict) -> np.ndarray:
        """Return an RGB uint8 image (res x res x 3) for one view. `data` holds this
        stage's outputs and every earlier one."""
        raise NotImplementedError

    def detail(self, ctx: Context, data: dict, p: dict, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        """Heights (m) at world coordinates finer than the preview grid (the detail
        window): the macro result sampled there plus the stage's micro relief."""
        raise NotImplementedError

    def stats(self, ctx: Context, data: dict) -> dict:
        """Short numbers shown under the map (optional)."""
        return {}
