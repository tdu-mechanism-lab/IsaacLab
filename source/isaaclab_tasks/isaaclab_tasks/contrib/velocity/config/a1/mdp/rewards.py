# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the A1-based dog and horse models (``calf2`` joints)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def feet_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
) -> torch.Tensor:
    """Reward lifting the feet above a target height.

    The height is measured in the world frame, so this term is only meaningful on flat ground.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # foot height (num_envs, num_feet)
    foot_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    clearance = torch.clamp(foot_height - target_height, min=0.0)
    return torch.mean(clearance, dim=1)


def feet_clearance_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward foot clearance only for feet that are in the swing phase.

    A foot is in swing when its contact force stayed below ``contact_threshold`` over the whole sensor history.
    The mean and the minimum over the feet are averaged, which discourages lifting a single foot only.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    net_forces = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    contact_force_mag = torch.norm(net_forces, dim=-1).max(dim=1)[0]  # (num_envs, num_feet)
    is_swing = (contact_force_mag < contact_threshold).float()
    per_foot_reward = torch.clamp(foot_height - target_height, min=0.0) * is_swing
    return 0.5 * torch.mean(per_foot_reward, dim=1) + 0.5 * torch.min(per_foot_reward, dim=1).values


def calf2_joint_motion(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward the motion (mean absolute velocity) of the ``calf2`` joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.mean(torch.abs(joint_vel), dim=1)


def joint_torques_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize applied joint torques with a per-joint weight (L2 squared kernel).

    Args:
        joint_weights: Mapping from joint name to weight. Joints that are not listed (and all joints when
            this is None) use a weight of 1.0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.actuators.applied_effort.torch  # (num_envs, num_joints)
    weights = joint_weights or {}
    w = torch.tensor([weights.get(name, 1.0) for name in asset.joint_names], device=torques.device, dtype=torques.dtype)
    return torch.sum(torch.square(torques) * w, dim=1)


def hip_inward_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """Penalize hip joints that tilt the legs inward (toward the body midline) beyond a threshold (L2 squared kernel).

    The hip axes of the models point forward, so a leg tilts inward at a negative hip angle on the left side and at a
    positive one on the right side.

    Args:
        threshold: Inward hip angle [rad] that is not penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    inward_sign = torch.tensor(
        [1.0 if name[1] == "R" else -1.0 for name in joint_names], device=joint_pos.device, dtype=joint_pos.dtype
    )
    excess = torch.clamp(joint_pos * inward_sign - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=1)
