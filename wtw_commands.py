"""WTW command contract and keyboard state, independent of the training repo."""

import numpy as np

COMMAND_NAMES = (
    "vx",
    "vy",
    "yaw_rate",
    "height",
    "frequency",
    "phase",
    "offset",
    "bound",
    "duration",
    "swing_height",
    "pitch",
    "roll",
    "width",
    "length",
    "aux",
)
COMMAND_SCALE = (
    2.0,
    2.0,
    0.25,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.15,
    0.3,
    0.3,
    1.0,
    1.0,
    1.0,
)
GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
}


def validate_command(command):
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (15,) or not np.isfinite(command).all():
        raise ValueError("WTW requires 15 finite commands")
    if command[4] <= 0 or not 0.1 <= command[8] <= 0.9:
        raise ValueError(
            "WTW frequency must be positive; duty factor must be in [0.1, 0.9]"
        )
    if command[9] < 0 or np.any(command[12:14] <= 0) or command[14] != 0:
        raise ValueError("Invalid WTW swing height, stance dimensions, or reserved aux")
    return command


def initial_command(config, geometry):
    settings = config.get("wtw", {})
    command = np.zeros(15, dtype=np.float32)
    command[:3] = config.get("cmd_init", [0.0, 0.0, 0.0])
    command[5:8] = GAITS[settings.get("gait", "trot")]
    for index, name, default in (
        (3, "height", 0.0),
        (4, "frequency", 2.5),
        (8, "duration", 0.5),
        (9, "swing_height", 0.06),
        (10, "pitch", 0.0),
        (11, "roll", 0.0),
        (12, "width", geometry["stance_width"]),
        (13, "length", geometry["stance_length"]),
    ):
        command[index] = settings.get(name, default)
    return validate_command(command)


class WTWCommandState:
    """Persistent gait/body settings; held movement keys are handled separately."""

    def __init__(self, command):
        self.defaults = validate_command(command).copy()
        self.command = self.defaults.copy()
        self.gait = next(
            (k for k, v in GAITS.items() if np.allclose(self.command[5:8], v)), "custom"
        )
        width, length = map(float, self.defaults[12:14])
        # Body/frequency ranges match the trained command curriculum.
        self.adjustments = {
            "r": (4, 0.1, 1.8, 3.2),
            "f": (4, -0.1, 1.8, 3.2),
            "t": (3, 0.01, -0.04, 0.04),
            "g": (3, -0.01, -0.04, 0.04),
            "y": (9, 0.01, 0.03, 0.09),
            "h": (9, -0.01, 0.03, 0.09),
            "u": (10, 0.025, -0.15, 0.15),
            "j": (10, -0.025, -0.15, 0.15),
            "i": (11, 0.025, -0.15, 0.15),
            "k": (11, -0.025, -0.15, 0.15),
            "o": (12, 0.005, width - 0.025, width + 0.025),
            "l": (12, -0.005, width - 0.025, width + 0.025),
            "p": (13, 0.005, length - 0.025, length + 0.025),
            ";": (13, -0.005, length - 0.025, length + 0.025),
            "]": (8, 0.025, 0.4, 0.6),
            "[": (8, -0.025, 0.4, 0.6),
        }

    def handle(self, key):
        if key in ("1", "2", "3", "4"):
            self.gait = tuple(GAITS)[int(key) - 1]
            self.command[5:8] = GAITS[self.gait]
        elif key == "0":
            self.command[:] = self.defaults
            self.gait = next(
                (k for k, v in GAITS.items() if np.allclose(self.defaults[5:8], v)),
                "custom",
            )
        elif key in self.adjustments:
            index, step, low, high = self.adjustments[key]
            self.command[index] = np.clip(float(self.command[index]) + step, low, high)
        elif key == "space":
            self.command[:3] = 0
        validate_command(self.command)
