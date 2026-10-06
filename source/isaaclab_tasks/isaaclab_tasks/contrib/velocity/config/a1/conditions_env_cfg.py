# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity-tracking environments with non-terrain conditions (payload, push, low friction).

* ``UnitreeA1ConditionsTrainEnvCfg``: training on the default Rough terrains. A payload level and a push level
  are assigned per environment (see ``conditions.py``), so one policy sees every combination.
* ``UnitreeA1Cond<Preset>EvalEnvCfg``: one environment per entry of ``CONDITION_PRESETS``. The evaluation
  protocol (fixed forward speed, no randomization) is applied on the custom terrain grid.
"""

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .conditions import CONDITION_PRESETS, apply_eval_protocol, payload_event, push_event, required_num_envs
from .custom_terrain_env_cfg import UnitreeA1CustomTerrainsTrainEnvCfg
from .rough_env_cfg import UnitreeA1RoughEnvCfg

A1_TRUNK_BODY = "trunk"
"""Name of the A1 trunk body that carries the payload."""

PAYLOAD_FRACTIONS = (0.0, 0.1, 0.25)
"""Extra trunk mass as a fraction of the robot mass, one level per group of environments."""

PUSH_FROUDE_LEVELS = (0.0, 0.15, 0.3)
"""Non-dimensional push strength (velocity change / sqrt(g * l_ref)), one level per group of environments."""


@configclass
class UnitreeA1ConditionsTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Rough-terrain training with payload and push levels spread over the environments."""

    def __post_init__(self):
        super().__post_init__()

        self.events.payload = payload_event(PAYLOAD_FRACTIONS, stride=1, body_name=A1_TRUNK_BODY)
        self.events.push_fixed = push_event(
            PUSH_FROUDE_LEVELS,
            push_time_s=5.0,
            direction="lateral",
            stride=len(PAYLOAD_FRACTIONS),
            sign_stride=len(PAYLOAD_FRACTIONS) * len(PUSH_FROUDE_LEVELS),
        )


@configclass
class UnitreeA1ConditionsPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_conditions"
    max_iterations = 3000


def _make_eval_env_cfg_class(preset: str):
    cond = CONDITION_PRESETS[preset]

    def __post_init__(self):
        UnitreeA1CustomTerrainsTrainEnvCfg.__post_init__(self)
        gen = self.scene.terrain.terrain_generator
        num_cells = gen.num_rows * gen.num_cols
        apply_eval_protocol(self, cond, num_cells=num_cells, payload_body_name=A1_TRUNK_BODY)
        self.scene.num_envs = required_num_envs(
            num_cells, len(cond.payload_fractions) or 1, len(cond.push_froude_levels) or 1
        )

    name = "UnitreeA1Cond" + "".join(part.capitalize() for part in preset.replace(".", "_").split("_")) + "EvalEnvCfg"
    cls = type(name, (UnitreeA1CustomTerrainsTrainEnvCfg,), {"__post_init__": __post_init__, "__module__": __name__})
    return name, configclass(cls)


EVAL_PRESET_CLASSES: dict[str, str] = {}
"""Preset name -> generated environment class name (all generated classes are module attributes)."""

for _preset in CONDITION_PRESETS:
    _name, _cls = _make_eval_env_cfg_class(_preset)
    globals()[_name] = _cls
    EVAL_PRESET_CLASSES[_preset] = _name

del _preset, _name, _cls
