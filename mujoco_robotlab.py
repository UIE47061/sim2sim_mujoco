import argparse
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import yaml

from keyboard_controller import KeyboardController


NUM_MOTOR = 12
DEFAULT_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]


def get_activation(act_name: str) -> nn.Module:
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"invalid activation function: {act_name}")


class Actor(nn.Module):
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def quat_rotate_inverse(q, v):
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded

    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0

    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0

    return a - b + c


def get_gravity_orientation(quaternion):
    quaternion = np.array(quaternion)
    gravity_world = np.array([0, 0, -1])

    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4)
        gravity_world = gravity_world.reshape(1, 3)
        result = quat_rotate_inverse(quaternion, gravity_world)[0]
    else:
        gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
        result = quat_rotate_inverse(quaternion, gravity_world)

    return result


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def infer_hidden_dims(actor_state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    hidden_dims = []
    layer_index = 0
    while f"mlp.{layer_index}.weight" in actor_state_dict:
        weight = actor_state_dict[f"mlp.{layer_index}.weight"]
        next_layer = layer_index + 2
        if f"mlp.{next_layer}.weight" not in actor_state_dict:
            break
        hidden_dims.append(weight.shape[0])
        layer_index = next_layer
    return tuple(hidden_dims)


def build_action_scale(action_scale_config) -> np.ndarray:
    if isinstance(action_scale_config, dict):
        return np.array(
            [action_scale_config["hip"]] * 4
            + [action_scale_config["thigh"]] * 4
            + [action_scale_config["calf"]] * 4,
            dtype=np.float32,
        )
    return np.full(NUM_MOTOR, action_scale_config, dtype=np.float32)


def get_actuator_joint_order(model: mujoco.MjModel) -> list[str]:
    joint_order = []
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_order.append(joint_name)
    return joint_order


def make_index_map(src_order: list[str], dst_order: list[str]) -> np.ndarray:
    src_to_index = {name: index for index, name in enumerate(src_order)}
    missing = [name for name in dst_order if name not in src_to_index]
    if missing:
        raise ValueError(f"joint order is missing joints: {missing}")
    return np.array([src_to_index[name] for name in dst_order], dtype=np.int64)


def read_joint_positions(data: mujoco.MjData, model: mujoco.MjModel, joint_order):
    values = []
    for joint_name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        values.append(data.qpos[model.jnt_qposadr[joint_id]])
    return np.array(values, dtype=np.float32)


def read_joint_velocities(data: mujoco.MjData, model: mujoco.MjModel, joint_order):
    values = []
    for joint_name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        values.append(data.qvel[model.jnt_dofadr[joint_id]])
    return np.array(values, dtype=np.float32)


def set_joint_positions(data: mujoco.MjData, model: mujoco.MjModel, joint_order, positions):
    for joint_name, position in zip(joint_order, positions):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = position


def load_actor(policy_path: str, num_obs: int, num_actions: int) -> Actor:
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise KeyError(
            "RobotLab checkpoint must contain actor_state_dict. "
            "Use mujoco_dwaq.py for DWAQ checkpoints."
        )

    actor_state_dict = checkpoint["actor_state_dict"]
    inferred_num_obs = actor_state_dict["mlp.0.weight"].shape[1]
    if inferred_num_obs != num_obs:
        raise ValueError(
            f"config num_obs={num_obs}, but checkpoint expects {inferred_num_obs}"
        )

    hidden_dims = infer_hidden_dims(actor_state_dict)
    policy = Actor(
        num_obs=num_obs,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        activation="elu",
    )
    mlp_state_dict = {
        key: value for key, value in actor_state_dict.items() if key.startswith("mlp.")
    }
    policy.load_state_dict(mlp_state_dict, strict=False)
    policy.eval()

    print("[INFO] RobotLab actor loaded:")
    print(f"  - Policy path: {policy_path}")
    print(f"  - Actor input: {num_obs}")
    print(f"  - Hidden dims: {hidden_dims}")
    print(f"  - Actor output: {num_actions}")
    return policy


if __name__ == "__main__":
    keyboard = KeyboardController()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="YAML config file path")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = build_action_scale(config["action_scale"])
    clip_actions = config.get("clip_actions")
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    policy = load_actor(policy_path, num_obs, num_actions)

    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    policy_joint_order = config.get("joint_order", DEFAULT_JOINT_ORDER)
    actuator_joint_order = get_actuator_joint_order(model)
    policy_to_actuator = make_index_map(policy_joint_order, actuator_joint_order)

    set_joint_positions(data, model, policy_joint_order, default_angles)
    mujoco.mj_forward(model, data)

    print("[INFO] Joint mapping:")
    for actuator_index, policy_index in enumerate(policy_to_actuator):
        print(
            f"  ctrl[{actuator_index:02d}] {actuator_joint_order[actuator_index]} "
            f"<- policy[{policy_index:02d}] {policy_joint_order[policy_index]}"
        )

    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    feet_list = []

    counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            qpos_actuator_order = read_joint_positions(data, model, actuator_joint_order)
            qvel_actuator_order = read_joint_velocities(data, model, actuator_joint_order)
            tau = pd_control(
                target_dof_pos[policy_to_actuator],
                qpos_actuator_order,
                kps[policy_to_actuator],
                np.zeros(NUM_MOTOR),
                qvel_actuator_order,
                kds[policy_to_actuator],
            )
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            counter += 1
            if counter % control_decimation == 0:
                foot_ids = [4, 6, 8, 10]
                foot_heights = [data.xipos[i][2] for i in foot_ids]

                qpos = read_joint_positions(data, model, policy_joint_order)
                qvel = read_joint_velocities(data, model, policy_joint_order)
                imu_quat = data.sensordata[36:40]
                ang_vel_b = data.sensordata[40:43]
                lin_vel_b = data.sensordata[49:52]
                gravity_b = get_gravity_orientation(imu_quat)

                cmd = keyboard.read()
                obs_list = [
                    ang_vel_b * ang_vel_scale,
                    gravity_b,
                    cmd * cmd_scale,
                    (qpos - default_angles) * dof_pos_scale,
                    qvel * dof_vel_scale,
                    action.astype(np.float32),
                ]
                obs = np.concatenate(obs_list, axis=0).astype(np.float32)
                if obs.shape[0] != num_obs:
                    raise RuntimeError(
                        f"built obs has size {obs.shape[0]}, but config num_obs={num_obs}"
                    )

                lin_vel_data_list.append(lin_vel_b * lin_vel_scale)
                ang_vel_data_list.append(ang_vel_b * ang_vel_scale)
                gravity_b_list.append(gravity_b)
                joint_vel_list.append(qvel * dof_vel_scale)
                action_list.append(action)
                feet_list.append(foot_heights)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_tensor = policy(obs_tensor)
                action = action_tensor.squeeze(0).numpy()
                if clip_actions is not None:
                    action = np.clip(action, -clip_actions, clip_actions)

                if counter < 300:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if not feet_list:
        raise RuntimeError("simulation ended before any policy step was collected")

    plt.figure(figsize=(18, 20))

    plt.subplot(4, 2, 1)
    for i in range(3):
        plt.plot([step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.title("History Linear Velocity", fontsize=10, pad=10)
    plt.legend()

    plt.subplot(4, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.title("History Angular Velocity", fontsize=10, pad=10)
    plt.legend()

    plt.subplot(4, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title("History Project Gravity", fontsize=10, pad=10)
    plt.legend()

    plt.subplot(4, 2, 5)
    for i in range(2):
        plt.plot([step[i] for step in joint_vel_list], label=f"Joint Velocity {i}")
    plt.title("History Joint Velocity", fontsize=10, pad=10)
    plt.legend()

    plt.subplot(4, 2, 6)
    for i in range(2):
        plt.plot([step[i] for step in action_list], label=f"Action {i}")
    plt.title("History Action", fontsize=10, pad=10)
    plt.legend()

    plt.subplot(4, 2, 7)
    foot_names = ["FL", "FR", "RL", "RR"]
    for i in range(len(feet_list[0])):
        plt.plot([step[i] for step in feet_list], label=f"{foot_names[i]} Height")
    plt.title("Foot Height (z)", fontsize=10, pad=10)
    plt.legend()

    plt.tight_layout()
    plt.show()
