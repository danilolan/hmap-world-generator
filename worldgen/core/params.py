"""Parameter declarations. Each stage lists its knobs as Param objects; the UI
builds its sliders, checkboxes and dropdowns from them."""
from dataclasses import dataclass, field


@dataclass
class Param:
    key: str
    label: str
    default: object
    kind: str = "float"          # float | int | bool | choice
    min: float = None
    max: float = None
    step: float = None
    choices: list = field(default_factory=list)
    help: str = ""
    advanced: bool = False       # shown under the stage's collapsed "Advanced" section

    def to_json(self) -> dict:
        return dict(key=self.key, label=self.label, default=self.default, kind=self.kind, min=self.min,
                    max=self.max, step=self.step, choices=self.choices, help=self.help,
                    advanced=self.advanced)

    def coerce(self, value):
        """Clamp and convert a value coming from the UI; fall back to the default."""
        if value is None:
            return self.default
        try:
            if self.kind == "bool":
                return bool(value)
            if self.kind == "choice":
                return value if value in self.choices else self.default
            v = float(value)
            if self.min is not None:
                v = max(self.min, v)
            if self.max is not None:
                v = min(self.max, v)
            return int(round(v)) if self.kind == "int" else v
        except (TypeError, ValueError):
            return self.default


def Float(key, label, default, lo, hi, step=None, help="", advanced=False):
    return Param(key, label, float(default), "float", lo, hi, step or (hi - lo) / 200.0, help=help, advanced=advanced)


def Int(key, label, default, lo, hi, step=1, help="", advanced=False):
    return Param(key, label, int(default), "int", lo, hi, step, help=help, advanced=advanced)


def Bool(key, label, default, help="", advanced=False):
    return Param(key, label, bool(default), "bool", help=help, advanced=advanced)


def Choice(key, label, default, choices, help="", advanced=False):
    return Param(key, label, default, "choice", choices=list(choices), help=help, advanced=advanced)
