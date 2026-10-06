# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity-tracking environments on the user-defined terrains of ``custom_terrains.py``.

* ``UnitreeA1CustomTerrainsTrainEnvCfg``: terrain curriculum over the 10 custom terrain types (columns) and
  10 difficulty levels (rows). The terrains vary along +x only, so the robot spawns facing +x and the command
  is forward-only.
* ``UnitreeA1CustomTerrainsEvalEnvCfg``: curriculum disabled, a fixed forward speed and no randomization.
  Pin the environments to the grid cells with ``custom_terrains.pin_envs_to_cells`` in an evaluation script.

Rewards, actions, observations and terminations are inherited from ``UnitreeA1RoughEnvCfg``; only the terrain
(and the command ranges) are replaced.
"""

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .conditions import EvalCondition, apply_eval_protocol
from .custom_terrains import (
    HfCorrugationTerrainCfg,
    HfRoughFieldTerrainCfg,
    HfSlope1DTerrainCfg,
    HfStep1DTerrainCfg,
    MeshBeam1DTerrainCfg,
    MeshGap1DTerrainCfg,
    make_eval6_terrain_cfg,
)
from .eval_terrain_env_cfg import add_fall_termination
from .rough_env_cfg import UnitreeA1RoughEnvCfg

NUM_ROWS = 10


def make_a1_custom_terrain_cfg(num_rows: int = NUM_ROWS):
    """The custom terrain grid with the ranges scaled down for the A1 (standing height ~0.3 m)."""
    gen = make_eval6_terrain_cfg(num_rows=num_rows, size=10.0, extras=True)
    for sub in gen.sub_terrains.values():
        if isinstance(sub, HfStep1DTerrainCfg):
            sub.step_height_range = (0.02, 0.12)
        elif isinstance(sub, HfCorrugationTerrainCfg):
            sub.amplitude = 0.05
        elif isinstance(sub, HfRoughFieldTerrainCfg):
            sub.rms_range = (0.0, 0.04)
        elif isinstance(sub, MeshGap1DTerrainCfg):
            sub.gap_width_range = (0.05, 0.25)
        elif isinstance(sub, MeshBeam1DTerrainCfg):
            sub.beam_width_range = (0.45, 0.15)
        elif isinstance(sub, HfSlope1DTerrainCfg):
            sub.slope_range = (sub.slope_range[0], 0.0 if sub.slope_range[1] == 0.0 else 0.4)
    return gen


@configclass
class UnitreeA1CustomTerrainsTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Training with the terrain-level curriculum on the custom terrains."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = make_a1_custom_terrain_cfg()
        self.scene.terrain.max_init_terrain_level = 5
        # the terrains only change along +x: spawn facing +x and command forward walking
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)}
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.ranges.lin_vel_x = (0.2, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)
        cmd.rel_standing_envs = 0.0
        # the gap, beam and cliff-edge cells have no floor below the robot
        add_fall_termination(self)


@configclass
class UnitreeA1CustomTerrainsEvalEnvCfg(UnitreeA1CustomTerrainsTrainEnvCfg):
    """Evaluation on the custom terrains: fixed forward speed, no randomization, one environment per cell."""

    def __post_init__(self):
        super().__post_init__()

        gen = self.scene.terrain.terrain_generator
        cond = EvalCondition("custom_terrains", speed_froude=0.3, l_ref=0.4)
        apply_eval_protocol(self, cond, num_cells=gen.num_rows * gen.num_cols)
        self.scene.num_envs = gen.num_rows * gen.num_cols

    def play_mode(self):
        """Keep the full terrain grid (the velocity play mode shrinks it to 5 x 5 and enables pushes)."""
        super().play_mode()
        gen = self.scene.terrain.terrain_generator
        gen.num_rows = NUM_ROWS
        gen.num_cols = len(gen.sub_terrains)
        gen.curriculum = True
        self.scene.num_envs = gen.num_rows * gen.num_cols


@configclass
class UnitreeA1CustomTerrainsPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_custom_terrains"
    max_iterations = 3000
