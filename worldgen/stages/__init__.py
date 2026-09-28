"""The world pipeline, in order (WORLDGEN.md section 2). Add each new stage here.
Stage 0 (seed and scale) is the tool's top bar: the master seed and the fixed
64 km world side (Pipeline.world_m); every stage derives its own sub-seed."""
from .character import Character
from .climate import Climate
from .erosion import Erosion
from .hydrology import Hydrology
from .soil import Soil
from .biomes import Biomes
from .forests import Forests
from .coast import Coast
from .export import Export
from .land import Land
from .plates import Plates
from .relief import Relief

STAGES = [
    Plates(),      # stage 1
    Land(),        # stage 2
    Character(),   # stage 3
    Relief(),      # stage 4
    Erosion(),     # stage 5
    Climate(),     # stage 6
    Hydrology(),   # stage 7
    Soil(),        # stage 8
    Biomes(),      # stage 9
    Forests(),     # stage 10
    Coast(),       # stage 11
    Export(),      # stage 12
]
