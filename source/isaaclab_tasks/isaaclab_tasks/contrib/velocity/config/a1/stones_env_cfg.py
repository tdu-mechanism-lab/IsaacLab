# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity tracking on uneven terrain with movable stones.

Four cuboid stones (1, 2, 4 and 8 kg) are dropped around the robot at every reset. The robot does not see them
(the height scanner only ray-casts the terrain): it has to notice them through contact. Rewards, actions,
observations and terminations are inherited from ``UnitreeA1RoughEnvCfg``.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .rough_env_cfg import UnitreeA1RoughEnvCfg
from .stones import make_stone_collection, reset_stones

# sub-terrains kept for this task: the ones with small height changes, so that a stone dropped near the robot
# always lands on the ground
STONE_TERRAINS = ("boxes", "random_rough")


@configclass
class UnitreeA1StonesTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Training with movable stones on the boxes and random-rough terrains."""

    def __post_init__(self):
        super().__post_init__()

        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {k: gen.sub_terrains[k] for k in STONE_TERRAINS}
        for sub in gen.sub_terrains.values():
            sub.proportion = 1.0
        self.scene.stones = make_stone_collection()
        self.events.reset_stones = EventTerm(func=reset_stones, mode="reset")


@configclass
class UnitreeA1StonesPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_stones"
