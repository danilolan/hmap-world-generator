"""The game's angles of repose: the steepest slope each tile ground holds, as the height
difference allowed between neighbouring corners 2 m apart.

They are game rules, not landforms (Docs/Design/04 in the game repository: "the map
respects the same repose" players are bound by), so they are read from the game's own
tuning asset, Assets/MegaSurvival/Config/InteractionTuning.asset (Repose*Units, 0.1 m),
found through MEGASURVIVAL_GAME or the MegaSurvival folder beside this repository. The
defaults below are used when the game repository is not there.
"""
import os
import re
from pathlib import Path

import numpy as np

# per TileGround value (dirt, sand, clay, gravel, rock): metres over a 2 m edge
DEFAULT_M = (1.7, 1.1, 2.4, 1.4, 7.4)
_KEYS = ("ReposeDirtUnits", "ReposeSandUnits", "ReposeClayUnits", "ReposeGravelUnits", "ReposeRockUnits")


def tuning_asset():
    game = Path(os.environ.get("MEGASURVIVAL_GAME", Path(__file__).resolve().parents[3] / "MegaSurvival"))
    return game / "Assets" / "MegaSurvival" / "Config" / "InteractionTuning.asset"


def load():
    """(repose per ground in metres over 2 m, source description)."""
    f = tuning_asset()
    try:
        text = f.read_text(encoding="utf-8")
        units = [int(re.search(rf"^\s*{k}:\s*(\d+)", text, re.M).group(1)) for k in _KEYS]
        return np.array(units, np.float64) * 0.1, str(f)
    except (OSError, AttributeError):
        return np.array(DEFAULT_M), "defaults (the game's InteractionTuning.asset was not found)"


REPOSE_M, SOURCE = load()


def rock_slope():
    """Rock's repose as a slope (rise over run)."""
    return REPOSE_M[4] / 2.0
