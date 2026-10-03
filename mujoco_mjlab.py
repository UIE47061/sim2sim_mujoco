"""Run an exported MJLab flat walking policy in standalone MuJoCo."""

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


class MjlabSim2Sim:
    def __init__(self, config):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / config["xml_path"]))
        self.model.opt.timestep = config["simulation_dt"]
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(
            str(ROOT / config["policy_path"]), map_location="cpu"
        ).eval()
        joint_ids = [self.model.joint(name).id for name in config["joint_order"]]
        if len(set(joint_ids)) != 12 or self.model.nu != 12:
            raise ValueError("Expected 12 unique actuated joints")
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.mapping = np.array(
            [joint_ids.index(int(j)) for j in self.model.actuator_trnid[:, 0]]
        )
        self.default = np.asarray(config["default_angles"], dtype=float)
        self.kps = np.asarray(config["kps"], dtype=float)
        self.kds = np.asarray(config["kds"], dtype=float)
        if any(a.shape != (12,) for a in (self.default, self.kps, self.kds)):
            raise ValueError("Joint parameters must contain 12 entries")
        self.base_id = self.model.body("base_link").id
        self.action = np.zeros(12, dtype=np.float32)
        self.data.qpos[2] = config["initial_height"]
        self.data.qpos[self.qadr] = self.default
        mujoco.mj_forward(self.model, self.data)
        self.dt = config["simulation_dt"] * config["control_decimation"]
        self.max_torque = 0.0
        self.min_height = self.data.qpos[2]
        self.initial_position = self.data.qpos[:3].copy()
        print("[INFO] MJLab flat actor loaded; obs=48, action=12", flush=True)
        print("[INFO] policy -> actuator mapping:", self.mapping.tolist(), flush=True)

    def observation(self, command):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity = rotation.T @ self.data.sensor("frame_vel").data
        omega = self.data.sensor("imu_gyro").data.copy()
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate(
            (
                velocity,
                omega,
                gravity,
                self.data.qpos[self.qadr] - self.default,
                self.data.qvel[self.vadr],
                self.action,
                command,
            )
        ).astype(np.float32)
        if obs.shape != (self.config["num_obs"],) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid MJLab observation")
        return obs

    def advance(self, command):
        obs = self.observation(command)
        with torch.inference_mode():
            self.action = (
                self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            )
        if self.action.shape != (12,) or not np.isfinite(self.action).all():
            raise RuntimeError("Invalid actor output")
        target = self.default + self.config["action_scale"] * self.action
        for _ in range(self.config["control_decimation"]):
            torque = (
                self.kps * (target - self.data.qpos[self.qadr])
                - self.kds * self.data.qvel[self.vadr]
            )
            torque = np.clip(
                torque, -self.config["torque_limit"], self.config["torque_limit"]
            )
            self.data.ctrl[:] = torque[self.mapping]
            self.max_torque = max(self.max_torque, float(np.abs(torque).max()))
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("Nonfinite physics state")
        self.min_height = min(self.min_height, self.data.qpos[2])
        upright_cos = self.data.xmat[self.base_id].reshape(3, 3)[2, 2]
        return obs, upright_cos < np.cos(np.deg2rad(70))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        help="Fixed command; disables keyboard",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "output/mjlab_report.json"
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config_file).read_text())
    duration = (
        args.duration if args.duration is not None else config["simulation_duration"]
    )
    if duration <= 0:
        parser.error("Duration must be positive")
    torch.set_num_threads(1)
    sim = MjlabSim2Sim(config)
    keyboard = None
    command = np.array(
        args.command if args.command is not None else config["cmd_init"], dtype=float
    )
    if not args.headless and args.command is None:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from keyboard_controller import KeyboardController

        keyboard = KeyboardController()
    reason = "duration"

    def run(viewer=None):
        nonlocal reason, command
        while sim.data.time + 1e-8 < duration and (
            viewer is None or viewer.is_running()
        ):
            start = time.monotonic()
            if keyboard:
                command = keyboard.read()
            _, fallen = sim.advance(command)
            if viewer is not None:
                viewer.cam.lookat[:] = sim.data.xpos[sim.base_id]
                viewer.sync()
            if fallen:
                reason = "fell_over"
                print("[INFO] Robot fell over; playback stopped", flush=True)
                return
            if viewer is not None:
                time.sleep(max(0, sim.dt - (time.monotonic() - start)))
        if sim.data.time + 1e-8 < duration:
            reason = "viewer_closed"

    try:
        if args.headless:
            run()
        else:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
                run(viewer)
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        if keyboard:
            keyboard.p.terminate()
            keyboard.p.join(timeout=2)
            keyboard.q.close()
        report = {
            "task": "Mjlab-Velocity-Flat-LittleWhiteV3",
            "policy": config["policy_path"],
            "model": config["xml_path"],
            "simulated_seconds": round(sim.data.time, 4),
            "end_reason": reason,
            "last_command": command.tolist(),
            "displacement_m": (sim.data.qpos[:3] - sim.initial_position).tolist(),
            "min_base_height_m": float(sim.min_height),
            "max_abs_torque_nm": sim.max_torque,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
