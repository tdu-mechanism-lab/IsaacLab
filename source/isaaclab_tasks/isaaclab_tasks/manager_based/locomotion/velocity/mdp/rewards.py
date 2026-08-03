# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def feet_clearance(
        # version 1
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     target_height: float = 0.05,
# ) -> torch.Tensor:
#     """Reward the robot for lifting its feet above a target height."""

#     # robot articulation
#     asset = env.scene[asset_cfg.name]

#     # foot height (num_envs, num_feet)
#     foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

#     # reward for exceeding threshold
#     clearance = torch.clamp(foot_height - target_height, min=0.0)

#     # average across feet → (num_envs,)
#     return torch.mean(clearance, dim=1)

        # version 2
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     target_height: float = 0.05,
#     contact_threshold: float = 1.0,
# ) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]

#     # 足先の高さ
#     foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

#     # 接地力（接触センサーがある場合）
#     # contact = env.scene["contact_sensor"].data.net_forces_w[:, asset_cfg.body_ids, 2]
#     # swing_mask = (contact.abs() < contact_threshold).float()  # 非接地=スイング相

#     # 接触センサーがない場合：高さで判定
#     swing_mask = (foot_height > 0.01).float()  # 地面から少し上=スイング中

#     clearance = torch.clamp(foot_height - target_height, min=0.0)

#     # スイング相の脚のみ報酬
#     return torch.mean(clearance * swing_mask, dim=1)

        # version 3
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """接地していない脚（スイング相）にのみクリアランス報酬を与える。
    全脚が均等にスイングするよう、各脚ごとの報酬を均す項も加える。
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_cfg.name]

    # 足先の高さ
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # 接触力の履歴から接地判定（実際の物理接触で判定。高さに依存しない）
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force_mag = torch.norm(net_forces, dim=-1).max(dim=1)[0]  # (N, num_feet)
    is_swing = (contact_force_mag < contact_threshold).float()

    clearance = torch.clamp(foot_height - target_height, min=0.0)
    per_foot_reward = clearance * is_swing  # (N, num_feet)

    # 単純平均ではなく、各脚の寄与が均等になるよう
    # 「最も報酬の低い脚」にも報酬を与えやすくする（minを混ぜる）
    mean_reward = torch.mean(per_foot_reward, dim=1)
    min_reward  = torch.min(per_foot_reward, dim=1).values

    # meanとminの加重平均：1本だけ上げる戦略を抑制しつつ完全な厳格化を避ける
    return 0.5 * mean_reward + 0.5 * min_reward

# def feet_clearance_with_phase(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     target_height: float = 0.05,
#     phase_freq: float = 1.0,
# ) -> torch.Tensor:
#     """歩行位相に基づいてスイング相の脚に足先クリアランス報酬を与える"""

#     asset = env.scene[asset_cfg.name]
#     foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

#     # 歩行位相（0〜2π）
#     phase = env.episode_length_buf * env.step_dt * 2 * torch.pi * phase_freq

#     # ウォーク位相オフセット FR=0, FL=π/2, RR=π, RL=3π/2
#     offsets = torch.tensor(
#         [0.0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi],
#         device=env.device
#     )
#     phase_per_foot = phase.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 4)

#     # sin > 0 のときスイング相
#     swing_mask = (torch.sin(phase_per_foot) > 0).float()

#     clearance = torch.clamp(foot_height - target_height, min=0.0)
#     return torch.mean(clearance * swing_mask, dim=1)


def calf2_joint_motion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """calf2_jointの動きを促進する報酬"""

    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(joint_vel), dim=1)


def joint_torques_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_weights: dict | list | None = None,
) -> torch.Tensor:
    """
    Joint-wise weighted L2 penalty on applied torques.

    - joint_weights can be:
      * dict: { "FR_hip_joint": 1.0, "FR_thigh_joint": 0.5, ... }
      * list/tuple/torch.tensor: must match asset.data.joint_names order
      * None: fallback (uniform weights = 1.0)

    Returns (num_envs,) tensor (unweighted by the RewTerm.weight; RewardManager multiplies by weight).
    """
    asset = env.scene[asset_cfg.name]
    # applied torque (num_envs, num_joints)
    torques = asset.data.applied_torque  # torch.Tensor on device

    num_joints = torques.shape[1]

    # build weight vector (torch) aligned with joint ordering
    if joint_weights is None:
        w = torch.ones(num_joints, device=torques.device, dtype=torques.dtype)
    else:
        # if dict mapping names->scale
        if isinstance(joint_weights, dict):
            names = list(asset.data.joint_names)  # list of str
            vals = []
            for n in names:
                vals.append(joint_weights.get(n, 1.0))  # default 1.0 if not provided
            w = torch.tensor(vals, device=torques.device, dtype=torques.dtype)
        else:
            # list/tuple/tensor: assume same order as asset.data.joint_names
            w = torch.tensor(joint_weights, device=torques.device, dtype=torques.dtype)
            if w.numel() != num_joints:
                raise ValueError("joint_weights length does not match number of joints")

    # elementwise weighted square, then sum over joints -> (num_envs,)
    per_joint_sq = torch.square(torques) * w.unsqueeze(0)
    return torch.sum(per_joint_sq, dim=1)

