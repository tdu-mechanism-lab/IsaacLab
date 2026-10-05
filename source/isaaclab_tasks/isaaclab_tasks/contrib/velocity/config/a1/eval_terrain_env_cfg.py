# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity-tracking environments on single-type terrains.

Every terrain in ``TERRAIN_SPECS`` gets two environment classes and one runner class:

* ``UnitreeA1<Name>TrainEnvCfg``: curriculum enabled (difficulty increases row by row).
* ``UnitreeA1<Name>EvalEnvCfg`` : curriculum disabled, difficulty fixed (default 0.5), terrain seed fixed.
* ``UnitreeA1<Name>PPORunnerCfg``: same PPO settings as Rough, logs under ``unitree_a1_<snake_name>``.

Rewards, actions, observations, terminations, and events are all inherited from
``UnitreeA1RoughEnvCfg`` and are NOT redefined here. Only the terrain is replaced, so
reward values stay directly comparable with the Rough training runs.

All sizes are for the A1 (standing height ~0.3 m, stance width ~0.26 m).
Scale the ``*_range`` values up for larger morphologies (e.g. 0.4 m front leg length).
"""

from collections.abc import Callable

import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import UnitreeA1RoughPPORunnerCfg
from .rough_env_cfg import UnitreeA1RoughEnvCfg

# NOTE on ``convert_to_heightfield``:
#   It only affects backends that consume ``newton:heightfield:resolution`` (Newton).
#   PhysX always collides against the exact mesh. With Newton, True rasterizes the mesh
#   at ``horizontal_scale`` (0.1 m), which erases features thinner than that (rails, bars,
#   small objects) and cannot represent overhangs (floating ring). The mesh terrains below
#   therefore use False so the geometry is identical on every backend.

##
# Sub-terrain definitions
# difficulty (0 -> 1) linearly interpolates between the two ends of each *_range,
# except where noted (some ranges are traversed max -> min).
##


def wave_terrain() -> SubTerrainBaseCfg:
    """cos(y) + sin(x) undulation. Height within +/- amplitude, wavelength 2 m (4 waves over 8 m)."""
    return terrain_gen.HfWaveTerrainCfg(
        amplitude_range=(0.02, 0.10),
        num_waves=4,
        border_width=0.25,
    )


def stepping_stones_terrain() -> SubTerrainBaseCfg:
    """Square stones separated by holes.

    Stone width goes 0.60 -> 0.30 m, gap goes 0.10 -> 0.25 m with difficulty.
    Uses a 0.05 m grid (see TERRAIN_SPECS): sizes are truncated to whole grid cells, so at the
    default 0.1 m grid any gap below 0.1 m becomes 0 cells, i.e. a completely flat floor.
    """
    return terrain_gen.HfSteppingStonesTerrainCfg(
        stone_height_max=0.0,  # all stone tops at the same height
        stone_width_range=(0.30, 0.60),
        stone_distance_range=(0.10, 0.25),
        holes_depth=-0.5,  # default -10 m; -0.5 m keeps a fallen robot on the terrain
        platform_width=2.0,
        border_width=0.25,
    )


def rails_terrain() -> SubTerrainBaseCfg:
    """Two square rings of box rails to step over.

    rail_thickness_range is NOT sampled: [0] = inner rail thickness, [1] = outer rail thickness.
    Only the rail height depends on difficulty.
    """
    return terrain_gen.MeshRailsTerrainCfg(
        rail_thickness_range=(0.05, 0.10),
        rail_height_range=(0.02, 0.10),
        platform_width=2.0,
        convert_to_heightfield=False,  # 0.05 m rail is thinner than the 0.1 m grid
    )


def discrete_obstacles_terrain() -> SubTerrainBaseCfg:
    """Random rectangular pillars AND pits ("choice" mode picks from -h, -h/2, +h/2, +h)."""
    return terrain_gen.HfDiscreteObstaclesTerrainCfg(
        obstacle_height_mode="choice",
        obstacle_width_range=(0.3, 1.0),
        obstacle_height_range=(0.02, 0.10),
        num_obstacles=40,
        platform_width=2.0,
        border_width=0.25,
    )


def pit_terrain() -> SubTerrainBaseCfg:
    """The robot starts at the bottom of a square pit and has to climb one step out."""
    return terrain_gen.MeshPitTerrainCfg(
        pit_depth_range=(0.02, 0.12),
        platform_width=2.0,
        double_pit=False,
        convert_to_heightfield=False,
    )


def box_terrain() -> SubTerrainBaseCfg:
    """The robot starts on top of a square box and has to step down."""
    return terrain_gen.MeshBoxTerrainCfg(
        box_height_range=(0.02, 0.12),
        platform_width=2.0,
        double_box=False,
        convert_to_heightfield=False,
    )


def gap_terrain() -> SubTerrainBaseCfg:
    """A square gap surrounds the start platform. The gap has NO bottom: a robot that falls in is lost."""
    return terrain_gen.MeshGapTerrainCfg(
        gap_width_range=(0.05, 0.30),
        platform_width=2.0,
        convert_to_heightfield=False,
    )


def floating_ring_terrain() -> SubTerrainBaseCfg:
    """A ring floating above the ground around the platform (an overhang).

    Ring clearance goes 0.40 -> 0.25 m (max -> min) and ring width 0.3 -> 1.0 m with difficulty.
    At high difficulty the A1 trunk (~0.3 m) can hit the ring: it must crouch or step through.
    """
    return terrain_gen.MeshFloatingRingTerrainCfg(
        ring_width_range=(0.3, 1.0),
        ring_height_range=(0.25, 0.40),
        ring_thickness=0.05,
        platform_width=2.0,
        convert_to_heightfield=False,  # a heightfield cannot represent an overhang
    )


def star_terrain() -> SubTerrainBaseCfg:
    """Narrow bars radiating from a round center platform over bottomless holes.

    Bar width goes 0.80 -> 0.35 m (max -> min) with difficulty. A1 stance width is ~0.26 m.
    bar_height is the bar thickness (the space between bars has no floor).
    """
    return terrain_gen.MeshStarTerrainCfg(
        num_bars=6,
        bar_width_range=(0.35, 0.80),
        bar_height_range=(0.20, 0.20),
        platform_width=2.0,
        convert_to_heightfield=False,
    )


def repeated_pyramids_terrain() -> SubTerrainBaseCfg:
    """Randomly scattered cones. More, taller, and narrower with difficulty."""
    return terrain_gen.MeshRepeatedPyramidsTerrainCfg(
        object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
            num_objects=20, height=0.03, radius=0.40, max_yx_angle=0.0
        ),
        object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
            num_objects=60, height=0.12, radius=0.30, max_yx_angle=10.0
        ),
        platform_width=2.0,
        convert_to_heightfield=False,
    )


def repeated_boxes_terrain() -> SubTerrainBaseCfg:
    """Randomly scattered (optionally tilted) boxes. More, taller, and smaller with difficulty."""
    return terrain_gen.MeshRepeatedBoxesTerrainCfg(
        object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=20, height=0.03, size=(0.40, 0.40), max_yx_angle=0.0
        ),
        object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
            num_objects=60, height=0.12, size=(0.25, 0.25), max_yx_angle=10.0
        ),
        platform_width=2.0,
        convert_to_heightfield=False,
    )


def repeated_cylinders_terrain() -> SubTerrainBaseCfg:
    """Randomly scattered vertical cylinders (posts). More, taller, and thinner with difficulty."""
    return terrain_gen.MeshRepeatedCylindersTerrainCfg(
        object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
            num_objects=20, height=0.03, radius=0.20, max_yx_angle=0.0
        ),
        object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
            num_objects=60, height=0.12, radius=0.10, max_yx_angle=10.0
        ),
        platform_width=2.0,
        convert_to_heightfield=False,
    )


# CamelName -> (snake_name, factory, horizontal_scale [m])
# CamelName is used in class names and task IDs; snake_name in the sub-terrain key and log folder.
TERRAIN_SPECS: dict[str, tuple[str, Callable[[], SubTerrainBaseCfg], float]] = {
    "Wave": ("wave", wave_terrain, 0.1),
    "SteppingStones": ("stepping_stones", stepping_stones_terrain, 0.05),
    "Rails": ("rails", rails_terrain, 0.1),
    "DiscreteObstacles": ("discrete_obstacles", discrete_obstacles_terrain, 0.1),
    "Pit": ("pit", pit_terrain, 0.1),
    "Box": ("box", box_terrain, 0.1),
    "Gap": ("gap", gap_terrain, 0.1),
    "FloatingRing": ("floating_ring", floating_ring_terrain, 0.1),
    "Star": ("star", star_terrain, 0.1),
    "RepeatedPyramids": ("repeated_pyramids", repeated_pyramids_terrain, 0.1),
    "RepeatedBoxes": ("repeated_boxes", repeated_boxes_terrain, 0.1),
    "RepeatedCylinders": ("repeated_cylinders", repeated_cylinders_terrain, 0.1),
}


def make_single_terrain_generator(
    name: str,
    sub_cfg: SubTerrainBaseCfg,
    *,
    curriculum: bool,
    horizontal_scale: float = 0.1,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    num_rows: int = 10,
    num_cols: int = 20,
    seed: int | None = None,
) -> TerrainGeneratorCfg:
    """Build a terrain generator that contains only one sub-terrain type.

    Tile size, border, vertical scale, and slope threshold match ``ROUGH_TERRAINS_CFG``.
    The generator overwrites horizontal/vertical scale of height-field sub-terrains with its own values.
    """
    sub_cfg.proportion = 1.0
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=horizontal_scale,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=curriculum,
        difficulty_range=difficulty_range,
        seed=seed,
        sub_terrains={name: sub_cfg},
    )


def add_fall_termination(env_cfg) -> None:
    """End the episode when the base drops below -1.0 m (world z).

    Gap and Star have bottomless holes: a robot that falls in never touches anything, so the
    Rough terminations (time_out, base_contact) would let it fall for the rest of the episode.
    Terrain surfaces here stay above -0.5 m (stepping-stone holes) and the base sits ~0.3 m
    above the feet, so this term never fires on Rough or the other terrains.
    """
    env_cfg.terminations.fell = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -1.0}
    )


##
# Base classes
##


@configclass
class UnitreeA1SingleTerrainTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Training on a single terrain type with the terrain-level curriculum."""

    def terrain_key(self) -> str:
        """CamelName key in TERRAIN_SPECS. Overridden by the generated classes."""
        raise NotImplementedError

    def __post_init__(self):
        # Everything from Rough (robot, rewards, actions, terminations, events) is applied here
        super().__post_init__()

        # Replace only the terrain. The parent __post_init__ set generator.curriculum on the
        # OLD generator, so it is set explicitly here.
        name, factory, h_scale = TERRAIN_SPECS[self.terrain_key()]
        self.scene.terrain.terrain_generator = make_single_terrain_generator(
            name, factory(), curriculum=True, horizontal_scale=h_scale
        )
        self.scene.terrain.max_init_terrain_level = 5
        add_fall_termination(self)


@configclass
class UnitreeA1SingleTerrainEvalEnvCfg(UnitreeA1RoughEnvCfg):
    """Evaluation on a single terrain type at a fixed difficulty.

    Change the difficulty from the command line without editing this file:
        'env.scene.terrain.terrain_generator.difficulty_range=(0.8,0.8)'
    """

    eval_difficulty: float = 0.5
    terrain_seed: int = 0  # same seed -> identical terrain across morphologies

    def terrain_key(self) -> str:
        """CamelName key in TERRAIN_SPECS. Overridden by the generated classes."""
        raise NotImplementedError

    def __post_init__(self):
        # Rewards etc. are inherited from Rough unchanged (reward values stay comparable)
        super().__post_init__()

        name, factory, h_scale = TERRAIN_SPECS[self.terrain_key()]
        self.scene.terrain.terrain_generator = make_single_terrain_generator(
            name,
            factory(),
            curriculum=False,
            horizontal_scale=h_scale,
            difficulty_range=(self.eval_difficulty, self.eval_difficulty),
            num_rows=8,
            num_cols=8,
            seed=self.terrain_seed,
        )
        # spawn uniformly over the grid instead of by terrain level
        self.scene.terrain.max_init_terrain_level = None
        # stop the curriculum from moving robots between terrain levels during evaluation
        self.curriculum.terrain_levels = None
        add_fall_termination(self)

    def play_mode(self):
        """Called automatically by `isaaclab play` (after __post_init__, before Hydra overrides).

        The velocity base play_mode() forces num_rows=num_cols=5 and removes pushes.
        Keep the push removal, but restore our terrain grid.
        """
        super().play_mode()
        gen = self.scene.terrain.terrain_generator
        gen.num_rows = 8
        gen.num_cols = 8
        gen.curriculum = False


@configclass
class UnitreeA1AllTerrainsTrainEnvCfg(UnitreeA1RoughEnvCfg):
    """Training on ALL terrains at once (the 6 Rough sub-terrains + the 12 in TERRAIN_SPECS).

    Each terrain type gets 2 columns of the grid; difficulty increases along the rows (curriculum).
    Evaluate the resulting policy per terrain with the ``<Name>Eval`` tasks.

    The whole grid shares one horizontal_scale (0.1 m), so the stepping-stone gaps are
    widened to 0.10 -> 0.35 m (1 -> 3 grid cells) instead of using the 0.05 m grid.
    """

    include_rough: bool = True
    cols_per_terrain: int = 2

    def __post_init__(self):
        # Rough rewards etc. + A1-scaled Rough sub-terrains are applied here
        super().__post_init__()

        sub_terrains: dict[str, SubTerrainBaseCfg] = {}
        if self.include_rough:
            sub_terrains.update(self.scene.terrain.terrain_generator.sub_terrains)
        for key, (snake, factory, _h_scale) in TERRAIN_SPECS.items():
            sub_cfg = factory()
            if key == "SteppingStones":
                sub_cfg.stone_distance_range = (0.10, 0.35)
            sub_terrains[snake] = sub_cfg
        for sub_cfg in sub_terrains.values():
            sub_cfg.proportion = 1.0  # equal share of columns

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=self.cols_per_terrain * len(sub_terrains),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains=sub_terrains,
        )
        self.scene.terrain.max_init_terrain_level = 5
        add_fall_termination(self)


@configclass
class UnitreeA1AllTerrainsPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    experiment_name = "unitree_a1_all_terrains"
    max_iterations = 5000  # more terrain types -> slower curriculum progress than Rough (1500)


##
# Concrete classes, generated from TERRAIN_SPECS
#   UnitreeA1<Name>TrainEnvCfg / UnitreeA1<Name>EvalEnvCfg / UnitreeA1<Name>PPORunnerCfg
##


def _terrain_key_method(key: str):
    def terrain_key(self) -> str:
        return key

    return terrain_key


for _key, (_snake, _factory, _h_scale) in TERRAIN_SPECS.items():
    for _mode, _base in (("Train", UnitreeA1SingleTerrainTrainEnvCfg), ("Eval", UnitreeA1SingleTerrainEvalEnvCfg)):
        _cls_name = f"UnitreeA1{_key}{_mode}EnvCfg"
        _cls = type(_cls_name, (_base,), {"terrain_key": _terrain_key_method(_key), "__module__": __name__})
        globals()[_cls_name] = configclass(_cls)

    _runner_name = f"UnitreeA1{_key}PPORunnerCfg"
    _runner = type(
        _runner_name,
        (UnitreeA1ZRoughPPORunnerCfg,),
        {"__annotations__": {"experiment_name": str}, "experiment_name": f"unitree_a1_{_snake}", "__module__": __name__},
    )
    globals()[_runner_name] = configclass(_runner)

del _key, _snake, _factory, _h_scale, _mode, _base, _cls_name, _cls, _runner_name, _runner