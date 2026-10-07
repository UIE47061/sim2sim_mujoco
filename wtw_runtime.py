"""Standalone WTW v2 inference contract. See third_party/wtw for provenance."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wtw_commands import (
    COMMAND_NAMES,
    COMMAND_SCALE,
    GAITS,
    initial_command,
    validate_command,
)

JOINTS = [
    f"{leg}_{part}_joint"
    for leg in ("FR", "FL", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
DEFAULT = [0.0, 0.95, -1.7] * 4
FIELDS = [
    ["gravity", 3],
    ["commands", 15],
    ["q_relative", 12],
    ["dq", 12],
    ["last_action", 12],
    ["previous_action", 12],
    ["clock", 4],
]


def apply_grid_visuals(spec):
    """Restore the original scene's checker/edge texture after model validation."""
    import mujoco

    texture = spec.add_texture(
        name="wtw_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    material = spec.add_material(
        name="wtw_groundplane",
        texuniform=True,
        texrepeat=[5, 5],
        reflectance=0.2,
        rgba=[1, 1, 1, 1],
    )
    textures = list(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture.name
    material.textures = textures
    floor = next(g for g in spec.worldbody.geoms if g.name == "floor")
    floor.material = material.name
    floor.rgba = [1, 1, 1, 1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_export(config, root):
    policy = root / config["policy_path"]
    metadata = json.loads(policy.with_name("metadata.json").read_text())
    expected = {
        "schema_version": 2,
        "observation_version": 1,
        "gait_version": 2,
        "control_version": 1,
        "transform_version": 3,
        "obs_dim": 70,
        "history": 30,
        "action_dim": 12,
        "history_order": "oldest_to_newest",
        "obs_fields": FIELDS,
        "joints": JOINTS,
        "command_names": list(COMMAND_NAMES),
        "command_scale": list(COMMAND_SCALE),
        "gait_presets": {k: list(v) for k, v in GAITS.items()},
        "default": DEFAULT,
        "kp": 20.0,
        "kd": 0.5,
        "physics_dt": 0.005,
        "decimation": 4,
        "action_scale": 0.25,
        "action_clip": 4.0,
        "torque_limit": 27.0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Incompatible WTW export: {key}")
    if sha(policy) != metadata.get("policy_sha256"):
        raise ValueError("WTW policy SHA-256 mismatch")
    for key, expected_value in {
        "num_obs": 70,
        "history_length": 30,
        "num_actions": 12,
        "simulation_dt": 0.005,
        "control_decimation": 4,
        "action_scale": 0.25,
        "action_clip": 4.0,
        "torque_limit": 27.0,
        "kps": [20.0] * 12,
        "kds": [0.5] * 12,
        "default_angles": DEFAULT,
    }.items():
        value = np.asarray(config.get(key, np.nan))
        target = np.asarray(expected_value)
        if value.shape != target.shape or not np.allclose(
            value, target, atol=1e-8, rtol=0
        ):
            raise ValueError(f"WTW config must match training: {key}")
    if config["joint_order"] != JOINTS:
        raise ValueError("WTW requires FR/FL/RL/RR policy joint order")
    assets = (root / config["xml_path"]).parent
    manifest = json.loads((assets / "manifest.json").read_text())
    if (
        manifest.get("revision") != "0ab74ef5d6048345db7a541316019100416b57d6"
        or manifest.get("transform_version") != 3
    ):
        raise ValueError(
            "WTW requires the pinned Little White v3 source and transform v3"
        )
    if (root / config["xml_path"]).name != "scene.xml" or manifest != metadata[
        "assets"
    ]:
        raise ValueError("WTW model manifest/transform does not match policy")
    for name, digest in manifest["sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid WTW manifest path")
        if sha(assets / name) != digest:
            raise ValueError(f"WTW model SHA-256 mismatch: {name}")
    geometry = json.loads((assets / "geometry.json").read_text())
    if not np.isclose(
        config["initial_height"], geometry["initial_height"], rtol=0, atol=1e-8
    ):
        raise ValueError("WTW initial height must match nominal model geometry")
    return metadata, geometry


class WTWState:
    def __init__(self, config, root):
        self.metadata, self.geometry = validate_export(config, root)
        self.initial_command = initial_command(config, self.geometry)
        self.phase = torch.zeros(1)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.history = None

    def command(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape == (3,):
            full = self.initial_command.copy()
            full[:3] = command
            command = full
        return validate_command(command)

    def clock(self, command):
        c = torch.from_numpy(command)
        p, o, b = c[5], c[6], c[7]
        raw = torch.remainder(
            self.phase[:, None] + torch.stack((o, p + o + b, b, p)), 1.0
        )
        duty = c[8].clamp(0.1, 0.9)
        warped = torch.where(
            raw < duty, raw * 0.5 / duty, 0.5 + (raw - duty) * 0.5 / (1 - duty)
        )
        return torch.sin(2 * torch.pi * warped)

    def observation(self, rotation, q, dq, action, command):
        # Exactly the trained 70D order; no measured linear/angular velocity or privilege.
        obs = torch.cat(
            (
                torch.tensor(-rotation[2], dtype=torch.float32),
                torch.from_numpy(command) * torch.tensor(COMMAND_SCALE),
                torch.tensor(q, dtype=torch.float32) - torch.tensor(DEFAULT),
                torch.tensor(dq, dtype=torch.float32) * 0.05,
                torch.from_numpy(action),
                torch.from_numpy(self.previous_action),
                self.clock(command)[0],
            )
        ).clamp(-100.0, 100.0)
        if obs.shape != (70,) or not torch.isfinite(obs).all():
            raise RuntimeError("Invalid WTW observation")
        return obs.numpy()

    def append(self, obs):
        if self.history is None:
            self.history = np.zeros((30, 70), dtype=np.float32)
        self.history = np.concatenate((self.history[1:], obs[None]), axis=0)

    def reset(self):
        self.phase.zero_()
        self.previous_action[:] = 0
        self.history = None

    def finish_step(self, command, dt):
        self.phase = torch.remainder(self.phase + dt * torch.tensor([command[4]]), 1.0)
