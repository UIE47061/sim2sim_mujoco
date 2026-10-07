"""Run exported MJLab, DreamWaQ or WTW policies in standalone MuJoCo."""

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
        if any(key in config for key in ("terrain", "step_height", "step_width")):
            raise ValueError("Terrain generation settings are no longer supported; use xml_path.")
        self.config = config
        self.dreamwaq = config.get("policy_type", "mjlab") == "dreamwaq"
        self.wtw = config.get("policy_type", "mjlab") == "wtw"
        if config.get("policy_type", "mjlab") not in ("mjlab", "dreamwaq", "wtw"):
            raise ValueError("policy_type must be mjlab, dreamwaq or wtw")
        if self.wtw:
            from wtw_runtime import WTWState

            self.wtw_state = WTWState(config, ROOT)
        if self.dreamwaq:
            self.validate_dreamwaq_metadata()
            spec = mujoco.MjSpec.from_file(str(ROOT / config["xml_path"]))
            self.apply_dreamwaq_contacts(spec)
            self.model = spec.compile()
        elif self.wtw and config.get("show_grid", True):
            from wtw_runtime import apply_grid_visuals

            spec = mujoco.MjSpec.from_file(str(ROOT / config["xml_path"]))
            apply_grid_visuals(spec)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(str(ROOT / config["xml_path"]))
        self.model.opt.timestep = config["simulation_dt"]
        if self.dreamwaq:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            for name, value in {
                "impratio": 10.0,
                "iterations": 10,
                "tolerance": 1e-8,
                "ls_iterations": 20,
                "ls_tolerance": 0.01,
                "ccd_iterations": 50,
            }.items():
                if hasattr(self.model.opt, name):
                    setattr(self.model.opt, name, value)
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(
            str(ROOT / config["policy_path"]), map_location="cpu"
        ).eval()
        joint_ids = [self.model.joint(name).id for name in config["joint_order"]]
        if len(set(joint_ids)) != 12 or self.model.nu != 12:
            raise ValueError("Expected 12 unique actuated joints")
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.joint_ranges = self.model.jnt_range[joint_ids]
        self.mapping = np.array(
            [joint_ids.index(int(j)) for j in self.model.actuator_trnid[:, 0]]
        )
        if sorted(self.mapping.tolist()) != list(range(12)):
            raise ValueError("Expected exactly one actuator per policy joint")
        self.default = np.asarray(config["default_angles"], dtype=float)
        self.kps = np.asarray(config["kps"], dtype=float)
        self.kds = np.asarray(config["kds"], dtype=float)
        if any(a.shape != (12,) for a in (self.default, self.kps, self.kds)):
            raise ValueError("Joint parameters must contain 12 entries")
        self.base_id = self.model.body("base_link").id
        self.action = np.zeros(12, dtype=np.float32)
        self.history = None
        self.illegal_contact = False
        self.data.qpos[2] = config["initial_height"]
        self.data.qpos[self.qadr] = self.default
        mujoco.mj_forward(self.model, self.data)
        self.dt = config["simulation_dt"] * config["control_decimation"]
        self.max_torque = 0.0
        self.min_height = self.data.qpos[2]
        self.initial_position = self.data.qpos[:3].copy()
        label = "WTW student loaded; obs=70, history=[1,2100]" if self.wtw else (
            "DreamWaQ actor loaded; history=[1,6,45]"
            if self.dreamwaq
            else "MJLab flat actor loaded; obs=48"
        )
        print(f"[INFO] {label}, action=12", flush=True)
        print("[INFO] policy -> actuator mapping:", self.mapping.tolist(), flush=True)

    def apply_dreamwaq_contacts(self, spec):
        """Apply the training robot's contact profile to the shared scene.

        scene.xml also serves legacy policies. Its included robot inherits a
        1 mm margin and 1D foot contacts with friction 0.4; DreamWaQ was trained
        with zero margin, 3D foot contacts, friction 1, and self collisions.
        Apply before compilation so MuJoCo also builds the body collision masks
        and bounding volumes from these settings.
        """
        for body in spec.bodies:
            for geom in body.geoms:
                if body.name == spec.worldbody.name:
                    # Terrain shares the robot's XML defaults unless overridden.
                    if geom.contype or geom.conaffinity:
                        geom.margin = 0.0
                        geom.contype = 1
                        geom.conaffinity = 0
                        geom.condim = 3
                        geom.friction = (1.0, 0.005, 0.0005)
                    continue
                geom.margin = 0.0
                if geom.group != 3:
                    continue
                geom.contype = 1
                geom.conaffinity = 1
                geom.condim = 1
                geom.solref = (0.01, 1.0)
                if geom.name.endswith("_Foot_collision"):
                    geom.condim = 3
                    geom.friction = (1.0, 0.005, 0.0005)
                    geom.priority = 1
        print(
            "[INFO] DreamWaQ training contact settings applied to shared scene",
            flush=True,
        )

    def validate_dreamwaq_metadata(self):
        metadata_path = (ROOT / self.config["policy_path"]).with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "task_version": 2,
            "input_shape": ["batch", 6, 45],
            "output_shape": ["batch", 12],
            "command_frame": "body_yaw_velocity",
            "history_order": "oldest_to_newest",
            "history_reset": "repeat_first_observation",
            "normalization": "embedded",
            "observation_order": [
                "angular_velocity",
                "projected_gravity",
                "command",
                "joint_position_relative",
                "joint_velocity",
                "last_action",
            ],
            "joint_order": self.config["joint_order"],
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"Incompatible DreamWaQ metadata: {name}")
        if self.config["num_obs"] != 45 or self.config["num_actions"] != 12:
            raise ValueError("DreamWaQ requires 45 observations and 12 actions")
        pairs = {
            "default_angles": "default_angles",
            "kps": "kp",
            "kds": "kd",
            "simulation_dt": "physics_dt",
            "control_decimation": "control_decimation",
            "torque_limit": "torque_limit",
            "action_scale": "action_scale",
        }
        for config_key, metadata_key in pairs.items():
            if not np.allclose(
                self.config[config_key], metadata[metadata_key], rtol=0, atol=1e-8
            ):
                raise ValueError(f"DreamWaQ config must match metadata: {config_key}")

    def observation(self, command):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        if self.wtw:
            return self.wtw_state.observation(
                rotation, self.data.qpos[self.qadr], self.data.qvel[self.vadr],
                self.action, self.wtw_state.command(command),
            )
        if self.dreamwaq:
            obs = np.concatenate(
                (
                    rotation.T @ self.data.sensor("frame_ang_vel").data,
                    rotation.T @ np.array([0.0, 0.0, -1.0]),
                    command,
                    self.data.qpos[self.qadr] - self.default,
                    self.data.qvel[self.vadr],
                    self.action,
                )
            ).astype(np.float32)
            if obs.shape != (45,) or not np.isfinite(obs).all():
                raise RuntimeError("Invalid DreamWaQ observation")
            return obs
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
        if self.wtw:
            command = self.wtw_state.command(command)
        obs = self.observation(command)
        if self.wtw:
            if self.wtw_state.history is None:
                self.wtw_state.append(obs)
            policy_input = self.wtw_state.history.reshape(2100)
            self.wtw_state.previous_action = self.action.copy()
        elif self.dreamwaq:
            if self.history is None:
                self.history = np.repeat(obs[None], 6, axis=0)
            else:
                self.history = np.concatenate((self.history[1:], obs[None]), axis=0)
            policy_input = self.history
        else:
            policy_input = obs
        with torch.inference_mode():
            self.action = (
                self.policy(torch.from_numpy(policy_input).unsqueeze(0))
                .squeeze(0)
                .numpy()
            )
        if self.action.shape != (12,) or not np.isfinite(self.action).all():
            raise RuntimeError("Invalid actor output")
        target = self.default + self.config["action_scale"] * self.action
        if self.wtw:
            target = self.default + self.config["action_scale"] * np.clip(
                self.action, -self.config["action_clip"], self.config["action_clip"]
            )
        if self.dreamwaq:
            target = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self.illegal_contact = False
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
            if self.dreamwaq:
                for index, contact in enumerate(self.data.contact):
                    geoms = (int(contact.geom1), int(contact.geom2))
                    ground = any(self.model.geom_bodyid[g] == 0 for g in geoms)
                    trunk = any(
                        self.model.geom(g).name == "base_link_collision" for g in geoms
                    )
                    if ground and trunk:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, index, force)
                        self.illegal_contact |= bool(np.linalg.norm(force[:3]) > 1.0)
        mujoco.mj_forward(self.model, self.data)
        if self.wtw:
            self.wtw_state.finish_step(command, self.dt)
            self.wtw_state.append(self.observation(command))
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("Nonfinite physics state")
        self.min_height = min(self.min_height, self.data.qpos[2])
        upright_cos = self.data.xmat[self.base_id].reshape(3, 3)[2, 2]
        fallen = upright_cos < np.cos(np.deg2rad(70)) or self.illegal_contact
        if self.wtw:
            fallen = self.data.qpos[2] < .12 or upright_cos < .5
        return obs, fallen


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
    parser.add_argument("--report", type=Path)
    parser.add_argument("--policy", type=Path, help="Override exported policy.pt path")
    parser.add_argument("--gait", choices=("trot", "pace", "bound", "pronk"))
    parser.add_argument("--switches", action="store_true", help="WTW: switch gait every 5 seconds")
    for name in ("frequency", "height", "swing-height", "pitch", "roll", "width", "length", "duty"):
        parser.add_argument(f"--{name}", type=float, help="WTW command override")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config_file).read_text())
    if args.policy:
        config["policy_path"] = str(args.policy.resolve())
    wtw_options = ("gait", "frequency", "height", "swing_height", "pitch", "roll", "width", "length", "duty")
    if args.switches or any(getattr(args, key) is not None for key in wtw_options):
        if config.get("policy_type") != "wtw":
            parser.error("gait/body command options require WTW mode")
        settings = config.setdefault("wtw", {})
        for key in wtw_options:
            if getattr(args, key) is not None:
                settings["duration" if key == "duty" else key] = getattr(args, key)
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
    if sim.wtw:
        command = sim.wtw_state.command(command)
    if not args.headless and args.command is None and not args.switches:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        from keyboard_controller import KeyboardController

        keyboard = KeyboardController(
            **config.get("keyboard", {}),
            **({"wtw_commands": command.tolist()} if sim.wtw else {}),
        )
    reason = "duration"

    def run(viewer=None):
        nonlocal reason, command
        while sim.data.time + 1e-8 < duration and (
            viewer is None or viewer.is_running()
        ):
            start = time.monotonic()
            if keyboard:
                command = keyboard.read()
            if args.switches:
                from wtw_commands import GAITS

                command[5:8] = tuple(GAITS.values())[min(int((sim.data.time + 1e-8) / 5), 3)]
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
            keyboard.close()
        report = {
            "task": "WTW-LittleWhiteV3" if sim.wtw else "DreamWaQ-LittleWhiteV3"
            if sim.dreamwaq
            else "Mjlab-Velocity-Flat-LittleWhiteV3",
            "policy": config["policy_path"],
            "model": config["xml_path"],
            "simulated_seconds": round(sim.data.time, 4),
            "end_reason": reason,
            "last_command": command.tolist(),
            "displacement_m": (sim.data.qpos[:3] - sim.initial_position).tolist(),
            "min_base_height_m": float(sim.min_height),
            "max_abs_torque_nm": sim.max_torque,
            "mujoco_version": mujoco.__version__,
            "torch_version": torch.__version__,
        }
        if sim.wtw:
            report.update(
                command_names=sim.wtw_state.metadata["command_names"],
                selected_iteration=sim.wtw_state.metadata["iteration"],
                policy_sha256=sim.wtw_state.metadata["policy_sha256"],
                observation_dim=70, history_length=30,
                final_phase=float(sim.wtw_state.phase.item()),
            )
        report_path = args.report or ROOT / "output" / (
            "wtw_report.json" if sim.wtw else "dreamwaq_report.json" if sim.dreamwaq else "mjlab_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
