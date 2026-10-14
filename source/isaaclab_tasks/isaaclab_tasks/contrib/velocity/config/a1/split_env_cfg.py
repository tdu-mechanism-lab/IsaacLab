# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 training environments with a single terrain type or a single condition.

* ``UnitreeA1Rough<Name>TrainEnvCfg``: one of the 6 sub-terrains of the Rough terrain set, curriculum enabled.
* ``UnitreeA1Custom<Name>TrainEnvCfg``: one of the user-defined terrains of ``custom_terrains.py``, curriculum enabled.
* ``UnitreeA1Cond<Name>TrainEnvCfg``: the Rough terrains with a single condition (payload, push or friction).
  The default mass randomization and random push of Rough are disabled so that the condition is the only change.

The single-terrain environments for the terrains of ``TERRAIN_SPECS`` are defined in ``eval_terrain_env_cfg.py``.
Each environment has a PPO runner class that logs under its own experiment name.
"""

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .conditions import apply_friction_condition, payload_event, push_event
from .custom_terrain_env_cfg import UnitreeA1CustomTerrainsTrainEnvCfg
from .rough_env_cfg import UnitreeA1RoughEnvCfg

# CamelName -> sub-terrain key
ROUGH_TERRAINS: dict[str, str] = {
    "PyramidStairs": "pyramid_stairs",
    "PyramidStairsInv": "pyramid_stairs_inv",
    "Boxes": "boxes",
    "RandomRough": "random_rough",
    "HfPyramidSlope": "hf_pyramid_slope",
    "HfPyramidSlopeInv": "hf_pyramid_slope_inv",
}

CUSTOM_TERRAINS: dict[str, str] = {
    "SlopeUp": "slope_up",
    "SlopeDown": "slope_down",
    "StepUp": "step_up",
    "StepDown": "step_down",
    "Corrugation": "corrugation",
    "CrossSlope": "cross_slope",
    "RoughRms": "rough_rms",
    "Gap": "gap",
    "Beam": "beam",
}

# CamelName -> (kind, value): payload and push levels are spread over the environments, the friction is fixed.
CONDITIONS: dict[str, tuple[str, object]] = {
    "Payload": ("payload", (0.0, 0.1, 0.25, 0.5)),
    "PushLateral": ("push_lateral", (0.0, 0.15, 0.3, 0.45)),
    "PushBackward": ("push_backward", (0.0, 0.15, 0.3, 0.45)),
    "Friction08": ("friction", 0.8),
    "Friction06": ("friction", 0.6),
    "Friction04": ("friction", 0.4),
    "Friction02": ("friction", 0.2),
}

A1_TRUNK_BODY = "trunk"


@configclass
class UnitreeA1RoughSingleTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Training on a single sub-terrain of the Rough terrain set."""

    def terrain_key(self) -> str:
        """Sub-terrain key in the Rough terrain set. Overridden by the generated classes."""
        raise NotImplementedError

    def __post_init__(self):
        super().__post_init__()

        gen = self.scene.terrain.terrain_generator
        key = self.terrain_key()
        gen.sub_terrains = {key: gen.sub_terrains[key]}
        gen.sub_terrains[key].proportion = 1.0
        gen.curriculum = True


@configclass
class UnitreeA1CustomSingleTrainEnvCfg(UnitreeA1CustomTerrainsTrainEnvCfg):
    """Training on a single user-defined terrain."""

    def terrain_key(self) -> str:
        """Sub-terrain key in the custom terrain set. Overridden by the generated classes."""
        raise NotImplementedError

    def __post_init__(self):
        super().__post_init__()

        gen = self.scene.terrain.terrain_generator
        key = self.terrain_key()
        gen.sub_terrains = {key: gen.sub_terrains[key]}
        gen.sub_terrains[key].proportion = 1.0
        gen.num_cols = 10
        gen.curriculum = True


@configclass
class UnitreeA1ConditionSingleTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Rough-terrain training with a single condition and without the default mass and push randomization."""

    def condition_key(self) -> str:
        """CamelName key in CONDITIONS. Overridden by the generated classes."""
        raise NotImplementedError

    def __post_init__(self):
        super().__post_init__()

        # the condition below is the only disturbance
        self.events.push_robot = None
        self.events.add_base_mass = None
        kind, value = CONDITIONS[self.condition_key()]
        if kind == "payload":
            self.events.payload = payload_event(value, stride=1, body_name=A1_TRUNK_BODY)
        elif kind in ("push_lateral", "push_backward"):
            self.events.push_fixed = push_event(
                value, push_time_s=5.0, direction=kind.removeprefix("push_"), sign_stride=len(value)
            )
        elif kind == "friction":
            apply_friction_condition(self, value)
        else:
            raise ValueError(kind)


##
# Concrete classes, generated from the tables above
##


def _key_method(name: str, key: str):
    def method(self) -> str:
        return key

    method.__name__ = name
    return method


def _make_runner(name: str, experiment_name: str):
    cls = type(
        name,
        (UnitreeA1RoughPPORunnerCfg,),
        {"__annotations__": {"experiment_name": str}, "experiment_name": experiment_name, "__module__": __name__},
    )
    return configclass(cls)


for _camel, _key in ROUGH_TERRAINS.items():
    _cls = type(
        f"UnitreeA1Rough{_camel}TrainEnvCfg",
        (UnitreeA1RoughSingleTrainEnvCfg,),
        {"terrain_key": _key_method("terrain_key", _key), "__module__": __name__},
    )
    globals()[_cls.__name__] = configclass(_cls)
    globals()[f"UnitreeA1Rough{_camel}PPORunnerCfg"] = _make_runner(
        f"UnitreeA1Rough{_camel}PPORunnerCfg", f"unitree_a1_rough_{_key}"
    )

for _camel, _key in CUSTOM_TERRAINS.items():
    _cls = type(
        f"UnitreeA1Custom{_camel}TrainEnvCfg",
        (UnitreeA1CustomSingleTrainEnvCfg,),
        {"terrain_key": _key_method("terrain_key", _key), "__module__": __name__},
    )
    globals()[_cls.__name__] = configclass(_cls)
    globals()[f"UnitreeA1Custom{_camel}PPORunnerCfg"] = _make_runner(
        f"UnitreeA1Custom{_camel}PPORunnerCfg", f"unitree_a1_custom_{_key}"
    )

for _camel in CONDITIONS:
    _cls = type(
        f"UnitreeA1Cond{_camel}TrainEnvCfg",
        (UnitreeA1ConditionSingleTrainEnvCfg,),
        {"condition_key": _key_method("condition_key", _camel), "__module__": __name__},
    )
    globals()[_cls.__name__] = configclass(_cls)
    globals()[f"UnitreeA1Cond{_camel}PPORunnerCfg"] = _make_runner(
        f"UnitreeA1Cond{_camel}PPORunnerCfg", f"unitree_a1_cond_{_camel.lower()}"
    )

del _camel, _key, _cls
