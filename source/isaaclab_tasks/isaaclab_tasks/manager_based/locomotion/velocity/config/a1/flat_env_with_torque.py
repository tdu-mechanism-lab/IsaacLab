# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg


@configclass
class UnitreeA1FlatEnvWithTorqueCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        def add_torque_statistics(env):
		orig_get_statistics = env.get_statistics
		def new_get_statistics():
		    stats = orig_get_statistics()
		    torques = env.scene["robot"].data.applied_torque
		    stats["Torque/mean"] = torques.abs().mean().item()
		    stats["Torque/max"] = torques.abs().max().item()
		    return stats
		env.get_statistics = new_get_statistics

	    # ManagerBasedRLEnv の生成時に呼ばれる hook
	    self.env_hooks.append(add_torque_statistics)


class UnitreeA1FlatEnvWithTorqueCfg_PLAY(UnitreeA1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
