# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-tracking environments for the A1-based dog and horse models.

Both models have an extra ``calf2`` joint per leg (4 joints per leg). They share the Rough terrain setup of
``UnitreeA1RoughEnvCfg`` and differ in the robot asset and the default joint pose:

* Both use ``trunk_front`` / ``trunk_rear`` / ``head_neck`` bodies and spine and neck joints.
"""

import math

from isaaclab.envs.mdp import undesired_contacts
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import (  # isort: skip
    UNITREE_A1_DOG_CFG,
    UNITREE_A1_DOG_NOHEAD_CFG,
    UNITREE_A1_HORSE_CFG,
    UNITREE_A1_HORSE_NOHEAD_CFG,
)

from . import mdp
from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .rough_env_cfg import UnitreeA1RoughEnvCfg

# Per-joint weights of the torque penalty: hip torques are penalized 10x more than the other joints.
_LEGS = ("FR", "FL", "RR", "RL")
_JOINT_TORQUE_WEIGHTS = {
    f"{leg}_{joint}_joint": weight
    for leg in _LEGS
    for joint, weight in (("hip", 1.0), ("thigh", 0.1), ("calf", 0.1), ("foot", 0.1))
}


def _apply_calf_model_cfg(
    cfg: UnitreeA1RoughEnvCfg,
    robot_cfg,
    base_bodies: str,
    base_contact_bodies: str,
    undesired_contact_bodies: str,
) -> None:
    """Replace the A1 robot by a ``calf2`` model and retune rewards, events and terminations for it.

    Args:
        base_bodies: Regex of the trunk bodies whose mass is randomized.
        base_contact_bodies: Regex of the bodies that terminate the episode on contact.
        undesired_contact_bodies: Regex of the bodies that are penalized on contact.
    """
    # physics: the dog and horse USDs are authored for PhysX, so PhysX is the default backend
    cfg.sim.physics.default = cfg.sim.physics.isaacsim_physx
    # scene
    cfg.scene.robot = replace(robot_cfg, prim_path="{ENV_REGEX_NS}/Robot")
    # commands: forward-biased velocity commands, (almost) no yaw rate
    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (6.0, 10.0)
    cmd.ranges.lin_vel_x = (-0.5, 2.0)
    cmd.ranges.lin_vel_y = (-0.5, 0.5)
    cmd.ranges.ang_vel_z = (-0.1, 0.1)
    # events: do not scale the default joint pose, which is tuned for these models
    cfg.events.push_robot = None
    cfg.events.base_com = None
    cfg.events.add_base_mass.params["asset_cfg"].body_names = base_bodies
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    # rewards
    cfg.rewards.ang_vel_xy_l2.weight = -0.1
    cfg.rewards.flat_orientation_l2.weight = -2.5
    cfg.rewards.feet_air_time.params["threshold"] = 0.2
    cfg.rewards.feet_air_time.weight = 0.05
    cfg.rewards.track_lin_vel_xy_exp.weight = 3.0
    cfg.rewards.undesired_contacts = RewTerm(
        func=undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=undesired_contact_bodies),
            "threshold": 1.0,
        },
    )
    cfg.rewards.feet_torque = RewTerm(
        func=mdp.joint_torques_weighted,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_weights": _JOINT_TORQUE_WEIGHTS},
    )
    # terminations
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = base_contact_bodies


@configclass
class UnitreeA1DogRoughEnvCfg(UnitreeA1RoughEnvCfg):
    """3-link dog: front/rear trunk and a head, with spine and neck joints."""

    def __post_init__(self):
        super().__post_init__()

        _apply_calf_model_cfg(
            self,
            UNITREE_A1_DOG_CFG,
            base_bodies="trunk_.*",
            base_contact_bodies="(trunk_front|trunk_rear|head_neck|.*(hip|thigh|calf(?!2)).*)",
            undesired_contact_bodies=".*(trunk_front|trunk_rear|head_neck|HIP|THIGH|CALF)$",
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk_front"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk_front"
        # swing-phase foot clearance, using the contact sensor to detect swing
        self.rewards.feet_clearance = RewTerm(
            func=mdp.feet_clearance_swing,
            weight=0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "target_height": 0.05,
                "contact_threshold": 1.0,
            },
        )

    def play_mode(self):
        """Straight-line walking at the maximum commanded speed, for videos and quantitative evaluation."""
        super().play_mode()
        self.events.add_base_mass = None
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.resampling_time_range = (1.0e9, 1.0e9)
        cmd.ranges.lin_vel_x = (2.0, 2.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.debug_vis = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}


@configclass
class UnitreeA1HorseRoughEnvCfg(UnitreeA1DogRoughEnvCfg):
    """Horse: same body layout (front/rear trunk, head) and rewards as the dog, with its own asset and pose."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = replace(UNITREE_A1_HORSE_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def _apply_nohead(cfg: UnitreeA1RoughEnvCfg, robot_cfg) -> None:
    """Switch a dog configuration to a headless model: a single ``trunk`` body instead of front/rear/head bodies.

    The URDF-converted asset nests the bodies under ``Geometry/base_link``.
    """
    cfg.scene.robot = replace(robot_cfg, prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link/trunk"
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
    cfg.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = "(trunk|.*(hip|thigh|calf(?!2)).*)"
    cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*(trunk|HIP|THIGH|CALF)$"


@configclass
class UnitreeA1DogNoheadRoughEnvCfg(UnitreeA1DogRoughEnvCfg):
    """Headless dog (URDF-converted USD): a single ``trunk`` body, 16 leg joints."""

    def __post_init__(self):
        super().__post_init__()

        _apply_nohead(self, UNITREE_A1_DOG_NOHEAD_CFG)


@configclass
class UnitreeA1HorseNoheadRoughEnvCfg(UnitreeA1DogRoughEnvCfg):
    """Headless horse (URDF-converted USD): same layout and rewards as the headless dog, with the horse pose."""

    def __post_init__(self):
        super().__post_init__()

        _apply_nohead(self, UNITREE_A1_HORSE_NOHEAD_CFG)


@configclass
class UnitreeA1DogPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_dog"


@configclass
class UnitreeA1HorsePPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_horse"


@configclass
class UnitreeA1DogNoheadPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_dog_nohead"

    def __post_init__(self):
        super().__post_init__()

        # a learned scalar std can turn negative and abort the run ("normal expects all elements of std >= 0.0"),
        # which happened to the horse after 1265 iterations; the log parameterization cannot
        self.actor.distribution_cfg.std_type = "log"


@configclass
class UnitreeA1HorseNoheadPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_horse_nohead"

    def __post_init__(self):
        super().__post_init__()

        # a learned scalar std can turn negative and abort the run ("normal expects all elements of std >= 0.0"),
        # which happened to the horse after 1265 iterations; the log parameterization cannot
        self.actor.distribution_cfg.std_type = "log"
