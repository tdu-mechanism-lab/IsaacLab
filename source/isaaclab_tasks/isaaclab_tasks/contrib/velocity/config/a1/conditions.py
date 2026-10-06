# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""地形以外の条件（積載・外乱・低摩擦）と、評価用の環境設定パッチ（Isaac Lab 3.0、ManagerBasedRLEnv）

条件は 3 種類。
- 低摩擦: apply_friction_condition（設定を書き換え）。実行ごとに 1 値（地形の材料は全体で 1 つ）。
  指標の例はすべり量、成功率、CoT
- 積載: add_payload（startup のイベント項）。環境ごとに水準を割り当てる。指標の例は転倒率、CoT、ピッチ・ロール
- 外乱: push_fixed（interval のイベント項）。環境ごとに水準を割り当てる。指標の例は回復時間、転倒率、最大傾き

環境ごとの割り当ては「水準 = levels[(env_id // stride) % len(levels)]」。
custom_terrains.pin_envs_to_cells は env_id % (地形のマス数) でマスを決めるので、
stride = マス数 にすると、すべてのマス × すべての水準の組み合わせが同数ずつできる。
環境数は マス数 × 水準数 × 反復数 にする（required_num_envs）。

各環境に実際に適用した値は env.morph_conditions[名前]（長さ num_envs のテンソル）に記録する。
評価スクリプトはこれを環境ごとの結果と一緒に CSV に書き出す。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

G = 9.81


# ===========================================================================
# 共通
# ===========================================================================


def assign_levels(env_ids: torch.Tensor, levels: Sequence[float], stride: int = 1) -> torch.Tensor:
    """env_id ごとの水準。levels[(env_id // stride) % len(levels)]"""
    lv = torch.as_tensor(list(levels), dtype=torch.float32, device=env_ids.device)
    idx = torch.div(env_ids.long(), max(int(stride), 1), rounding_mode="floor") % len(levels)
    return lv[idx]


def required_num_envs(num_cells: int, *level_counts: int, reps: int = 1) -> int:
    """マス数 × 各条件の水準数 × 反復数。複数の条件を同時に掛け合わせる場合は stride も掛け合わせること"""
    n = num_cells * reps
    for c in level_counts:
        n *= c
    return n


def _log(env, name: str, env_ids: torch.Tensor, values: torch.Tensor):
    if not hasattr(env, "morph_conditions"):
        env.morph_conditions = {}
    buf = env.morph_conditions.get(name)
    if buf is None:
        buf = torch.full((env.num_envs,), float("nan"), device=values.device)
        env.morph_conditions[name] = buf
    buf[env_ids.long()] = values.to(buf.dtype)


def _all_env_ids(env, device) -> torch.Tensor:
    return torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)


# ===========================================================================
# 1. 積載（体幹への追加質量。ロボット総質量に対する割合で指定）
# ===========================================================================


class add_payload(ManagerTermBase):
    """体幹（base）に総質量 × fraction の質量を足す。mode="startup" で使う。

    - 質量は形態間で同じ総質量を前提に「総質量に対する割合」で決める（kg 指定より比較が公平）。
    - inertia_mode="point"：荷物を体幹の重心に置いた点質量とみなし、慣性は変えない。
      inertia_mode="scale"：体幹を一様密度とみなし、慣性を質量比で拡大する。
    - 体幹の重心位置は変えない（荷物の前後位置を変えたい場合は randomize_rigid_body_com を固定値で併用）。
    """

    def __init__(self, cfg: EventTerm, env):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self.default_mass = None
        self.default_inertia = None

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        fractions: Sequence[float],
        stride: int = 1,
        inertia_mode: str = "point",
    ):
        asset = self.asset
        if self.default_mass is None:
            self.default_mass = asset.data.body_mass.torch.clone()
            self.default_inertia = asset.data.body_inertia.torch.clone()
        if env_ids is None:
            env_ids = _all_env_ids(env, asset.device)
        env_ids = env_ids.to(asset.device)
        body_ids = asset_cfg.body_ids
        if body_ids == slice(None) or len(body_ids) != 1:
            raise ValueError("add_payload: asset_cfg は体幹 1 リンクだけを指定する（body_names='base' など）")
        body_ids_t = torch.tensor(body_ids, dtype=torch.int32, device=asset.device)

        frac = assign_levels(env_ids, fractions, stride)  # (n,)
        total = self.default_mass[env_ids.long()].sum(dim=1)  # (n,)
        dm = frac * total
        m0 = self.default_mass[env_ids.long()][:, body_ids_t.long()]  # (n, 1)
        m_new = m0 + dm[:, None]
        asset.set_masses_index(masses=m_new, body_ids=body_ids_t, env_ids=env_ids)
        if inertia_mode == "scale":
            I0 = self.default_inertia[env_ids.long()][:, body_ids_t.long()]  # (n, 1, 9)
            asset.set_inertias_index(inertias=I0 * (m_new / m0)[..., None], body_ids=body_ids_t, env_ids=env_ids)
        elif inertia_mode != "point":
            raise ValueError(f"inertia_mode must be 'point' or 'scale', got {inertia_mode!r}")
        _log(env, "payload_fraction", env_ids, frac)
        _log(env, "payload_kg", env_ids, dm)


def payload_event(fractions: Sequence[float], stride: int = 1, body_name: str = "base", inertia_mode: str = "point"):
    return EventTerm(
        func=add_payload,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=body_name),
            "fractions": tuple(fractions),
            "stride": stride,
            "inertia_mode": inertia_mode,
        },
    )


# ===========================================================================
# 2. 外乱（決まった時刻に、決まった大きさ・向きの速度変化を与える）
# ===========================================================================


def push_fixed(
    env,
    env_ids: torch.Tensor,
    froude_levels: Sequence[float],
    direction: str = "lateral",
    l_ref: float = 0.4,
    stride: int = 1,
    alternate_sign: bool = True,
    sign_stride: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """体幹の速度に Δv を足す（push_by_setting_velocity の決定版）。

    Δv = k × sqrt(g × l_ref)。k は froude_levels から環境ごとに割り当て（無次元の押しの強さ）。
    同じ総質量なら Δv をそろえることは力積をそろえることと同じ。
    direction：
      "lateral"  体幹の左向き（heading に垂直）。alternate_sign=True なら環境ごとに左右を交互にする
      "backward" 進行方向と逆
      "forward"  進行方向
    向きは heading（ヨー角）だけから決める（ピッチ・ロールの影響を受けない）。
    """
    asset = env.scene[asset_cfg.name]
    env_ids = env_ids.to(asset.device)
    k = assign_levels(env_ids, froude_levels, stride)
    dv = k * math.sqrt(G * l_ref)
    h = asset.data.heading_w.torch[env_ids.long()]
    if direction == "lateral":
        ux, uy = -torch.sin(h), torch.cos(h)
    elif direction == "backward":
        ux, uy = -torch.cos(h), -torch.sin(h)
    elif direction == "forward":
        ux, uy = torch.cos(h), torch.sin(h)
    else:
        raise ValueError(direction)
    if alternate_sign:
        sign = 1.0 - 2.0 * (torch.div(env_ids.long(), max(sign_stride, 1), rounding_mode="floor") % 2).float()
        dv = dv * sign
    vel = asset.data.root_vel_w.torch[env_ids.long()].clone()  # (n, 6) 線速度 xyz + 角速度 xyz（ワールド）
    vel[:, 0] += dv * ux
    vel[:, 1] += dv * uy
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
    _log(env, "push_dv", env_ids, dv)
    # 押した時刻（エピソード内のステップ数）を記録しておくと回復時間を計算しやすい
    _log(env, "push_step", env_ids, env.episode_length_buf[env_ids.long()].float())


def push_event(
    froude_levels: Sequence[float],
    push_time_s: float = 5.0,
    direction: str = "lateral",
    l_ref: float = 0.4,
    stride: int = 1,
    sign_stride: int = 1,
):
    """エピソード開始から push_time_s 秒ごとに押す（エピソード 20 s・5 s なら 5, 10, 15 s の 3 回）。
    1 回だけにしたいなら push_time_s をエピソード長の半分より大きくする。"""
    return EventTerm(
        func=push_fixed,
        mode="interval",
        interval_range_s=(push_time_s, push_time_s),
        is_global_time=False,
        resample_interval_on_reset=True,
        params={
            "froude_levels": tuple(froude_levels),
            "direction": direction,
            "l_ref": l_ref,
            "stride": stride,
            "sign_stride": sign_stride,
        },
    )


# ===========================================================================
# 3. 低摩擦（設定の書き換え。実行ごとに 1 値）
# ===========================================================================


def apply_friction_condition(env_cfg, mu: float, robot_body_names: str = ".*"):
    """地形とロボットの摩擦係数を同じ μ にそろえる。

    ねらい：組み合わせ規則（PhysX の multiply/average/min/max、MuJoCo 系の max）に依存せず、
    実効摩擦が μ になるようにする。ただし PhysX の multiply だけは μ² になるので、地形側を "min" にする。
    Newton（既定）では TerrainImporterCfg.physics_material の扱いと組み合わせ規則を必ず確認すること。
    """
    pm = env_cfg.scene.terrain.physics_material
    pm.static_friction = mu
    pm.dynamic_friction = mu
    if hasattr(pm, "friction_combine_mode"):
        pm.friction_combine_mode = "min"
    env_cfg.sim.physics_material = pm  # velocity_env_cfg の __post_init__ と同じ対応付け
    ev = getattr(env_cfg.events, "physics_material", None)
    if ev is not None:
        ev.params["asset_cfg"] = SceneEntityCfg("robot", body_names=robot_body_names)
        ev.params["static_friction_range"] = (mu, mu)
        ev.params["dynamic_friction_range"] = (mu, mu)
    env_cfg.morph_friction_mu = mu  # 記録用（評価スクリプトが CSV に書く）
    return env_cfg


# ===========================================================================
# 4. 評価用パッチ（乱数を止め、進行方向・速度を固定し、地形をマス固定にする）
# ===========================================================================


@dataclass
class EvalCondition:
    name: str = "baseline"
    speed_froude: float = 0.3
    """速度指令（前進）を無次元速度 v/sqrt(g·l_ref) で指定"""
    l_ref: float = 0.4
    friction_mu: float | None = None
    payload_fractions: tuple[float, ...] = ()
    push_froude_levels: tuple[float, ...] = ()
    push_time_s: float = 12.0
    push_direction: str = "lateral"
    episode_length_s: float = 20.0
    extra: dict = field(default_factory=dict)


def apply_eval_protocol(env_cfg, cond: EvalCondition, num_cells: int, payload_body_name: str = "base"):
    """__post_init__ 後の env_cfg に評価用の設定を上書きする。

    - 地形カリキュラムを止め、生成器は curriculum=True のまま（行が難易度順に並ぶ）
      ※ velocity_env_cfg.__post_init__ は curriculum 項が None だと生成器の curriculum も False にするので、
        この関数は __post_init__ の「後」に呼ぶ
    - 学習時の乱数（初期姿勢・関節・体幹質量・重心・押し）を止める
    - 速度指令を +x 方向の一定値に固定（heading=0）
    - 条件（摩擦・積載・外乱）を入れる（積載は payload_body_name のリンクに足す）
    環境をマスに固定する処理（pin_envs_to_cells）は、環境を作った後に評価スクリプト側で呼ぶ。
    """
    # 地形
    env_cfg.curriculum.terrain_levels = None
    gen = env_cfg.scene.terrain.terrain_generator
    if gen is not None:
        gen.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.episode_length_s = cond.episode_length_s

    # 観測ノイズを止める
    obs = getattr(env_cfg, "observations", None)
    if obs is not None and hasattr(obs, "policy"):
        obs.policy.enable_corruption = False

    # 乱数を止める
    ev = env_cfg.events
    ev.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    ev.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    ev.add_base_mass = None
    if hasattr(ev, "base_com"):
        ev.base_com = None
    ev.base_external_force_torque = None
    ev.push_robot = None

    # 速度指令
    v = cond.speed_froude * math.sqrt(G * cond.l_ref)
    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (v, v)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)
    cmd.heading_command = True
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 1.0
    cmd.resampling_time_range = (1e9, 1e9)

    # 条件
    if cond.friction_mu is not None:
        apply_friction_condition(env_cfg, cond.friction_mu)
    stride = num_cells
    if cond.payload_fractions:
        ev.payload = payload_event(cond.payload_fractions, stride=stride, body_name=payload_body_name)
        stride *= len(cond.payload_fractions)
    if cond.push_froude_levels:
        ev.push_fixed = push_event(
            cond.push_froude_levels,
            push_time_s=cond.push_time_s,
            direction=cond.push_direction,
            l_ref=cond.l_ref,
            stride=stride,
            sign_stride=stride * len(cond.push_froude_levels),
        )
    env_cfg.morph_eval_condition = cond
    return env_cfg


# よく使う条件の組（評価 1 回 = 1 条件）
CONDITION_PRESETS: dict[str, EvalCondition] = {
    "baseline": EvalCondition("baseline"),
    "friction_0.8": EvalCondition(
        "friction_0.8", friction_mu=0.8
    ),  # 摩擦系列の基準（地形とロボットを同じ μ にそろえた条件）
    "friction_0.6": EvalCondition("friction_0.6", friction_mu=0.6),
    "friction_0.4": EvalCondition("friction_0.4", friction_mu=0.4),
    "friction_0.2": EvalCondition("friction_0.2", friction_mu=0.2),
    "payload": EvalCondition("payload", payload_fractions=(0.0, 0.1, 0.25, 0.5)),
    "push_lateral": EvalCondition("push_lateral", push_froude_levels=(0.0, 0.15, 0.3, 0.45)),
    "push_backward": EvalCondition(
        "push_backward", push_froude_levels=(0.0, 0.15, 0.3, 0.45), push_direction="backward"
    ),
}
