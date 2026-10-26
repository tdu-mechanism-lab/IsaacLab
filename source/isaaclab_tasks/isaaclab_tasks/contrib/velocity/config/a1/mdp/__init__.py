# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms specific to the A1-based dog and horse models."""

from .rewards import calf2_joint_motion, feet_clearance, feet_clearance_swing, joint_torques_weighted
from .terminations import physics_diverged

__all__ = [
    "calf2_joint_motion",
    "feet_clearance",
    "feet_clearance_swing",
    "joint_torques_weighted",
    "physics_diverged",
]
