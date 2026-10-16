# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Movable stones (dynamic rigid bodies) scattered around the robot, and the event that re-places them."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# (edge length [m], mass [kg], color): from a light pebble to a heavy block the A1 (~12 kg) can hardly push
STONE_SPECS: tuple[tuple[float, float, tuple[float, float, float]], ...] = (
    (0.15, 1.0, (0.85, 0.55, 0.25)),
    (0.20, 2.0, (0.70, 0.45, 0.20)),
    (0.25, 4.0, (0.55, 0.35, 0.15)),
    (0.30, 8.0, (0.40, 0.25, 0.10)),
)


def make_stone_collection(specs=STONE_SPECS) -> RigidObjectCollectionCfg:
    """One cuboid per entry of ``specs``, spawned in every environment."""
    stones = {}
    for i, (size, mass, color) in enumerate(specs):
        stones[f"stone_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/stone_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(size, size, size),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
                    PhysxRigidBodyCfg(linear_damping=0.05, angular_damping=0.05, max_depenetration_velocity=1.0),
                ],
                mass_props=sim_utils.MassCfg(mass=mass),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            collision_group=0,
        )
    return RigidObjectCollectionCfg(rigid_objects=stones)


def reset_stones(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("stones"),
    radius_range: tuple[float, float] = (0.9, 2.5),
    drop_height: float = 0.6,
):
    """Drop every stone at a random spot of a ring around the environment origin, with a random yaw.

    The ring starts at ``radius_range[0]`` so a stone never overlaps the robot, which is reset within 0.5 m of the
    origin. The stones are dropped from ``drop_height`` above the terrain origin and settle by themselves.
    """
    stones = env.scene[asset_cfg.name]
    if isinstance(env_ids, slice):  # the first reset passes all environments as a slice
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()
    n, k = env_ids.numel(), stones.num_bodies
    r = torch.empty(n, k, device=env.device).uniform_(*radius_range)
    ang = torch.empty(n, k, device=env.device).uniform_(0.0, 2.0 * math.pi)
    yaw = torch.empty(n, k, device=env.device).uniform_(0.0, 2.0 * math.pi)
    pose = torch.zeros(n, k, 7, device=env.device)
    origins = env.scene.env_origins[env_ids]
    pose[..., 0] = origins[:, None, 0] + r * torch.cos(ang)
    pose[..., 1] = origins[:, None, 1] + r * torch.sin(ang)
    pose[..., 2] = origins[:, None, 2] + drop_height
    pose[..., 5] = torch.sin(0.5 * yaw)  # quaternion (x, y, z, w) of a rotation about z
    pose[..., 6] = torch.cos(0.5 * yaw)
    stones.write_body_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
    stones.write_body_velocity_to_sim_index(body_velocities=torch.zeros(n, k, 6, device=env.device), env_ids=env_ids)
