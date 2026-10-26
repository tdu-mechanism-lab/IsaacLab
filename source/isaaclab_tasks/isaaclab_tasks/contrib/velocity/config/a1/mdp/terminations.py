# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the A1-based dog and horse models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def physics_diverged(
    env: ManagerBasedRLEnv,
    max_joint_vel: float = 200.0,
    max_root_vel: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate the environments whose simulation blew up: a non-finite state or an unphysically fast joint or root.

    A single exploding environment otherwise feeds huge (but finite) observations and rewards into the policy
    update, which overflows the value loss and turns the network weights into NaN.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch
    root_vel = torch.cat((asset.data.root_lin_vel_w.torch, asset.data.root_ang_vel_w.torch), dim=1)
    non_finite = ~torch.isfinite(joint_vel).all(dim=1) | ~torch.isfinite(root_vel).all(dim=1)
    too_fast = (joint_vel.abs() > max_joint_vel).any(dim=1) | (root_vel.abs() > max_root_vel).any(dim=1)
    return non_finite | too_fast
