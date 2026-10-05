# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Flat-UnitreeA1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-UnitreeA1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

##
# Single-terrain environments, defined in eval_terrain_env_cfg.py (TERRAIN_SPECS).
# Rewards etc. are inherited from UnitreeA1RoughEnvCfg.
# Keep this list in sync with the keys of TERRAIN_SPECS. It is duplicated here on purpose:
# importing eval_terrain_env_cfg at registration time would pull in isaaclab.terrains.
##
 
_TERRAINS = (
    "Wave",
    "SteppingStones",
    "Rails",
    "DiscreteObstacles",
    "Pit",
    "Box",
    "Gap",
    "FloatingRing",
    "Star",
    "RepeatedPyramids",
    "RepeatedBoxes",
    "RepeatedCylinders",
)
 
for _terrain in _TERRAINS:
    for _mode in ("Train", "Eval"):
        gym.register(
            id=f"IsaacContrib-Velocity-{_terrain}{_mode}-UnitreeA1",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.eval_terrain_env_cfg:UnitreeA1{_terrain}{_mode}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.eval_terrain_env_cfg:UnitreeA1{_terrain}PPORunnerCfg",
            },
        )
 
# All terrains at once (6 Rough sub-terrains + the 12 above), for training a single generalist policy.
gym.register(
    id="IsaacContrib-Velocity-AllTerrainsTrain-UnitreeA1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.eval_terrain_env_cfg:UnitreeA1AllTerrainsTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.eval_terrain_env_cfg:UnitreeA1AllTerrainsPPORunnerCfg",
    },
)
 