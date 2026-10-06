# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""形態比較用の自作地形（Isaac Lab 3.0）

どの地形も「動かす量は 1 つ」「進行方向は +x」「x の両端と中心線 (y=0) で高さ 0」に統一してある。
評価では進行方向の指令を +x に固定し、環境をマスに固定して使う（pin_envs_to_cells）。

- HfSlope1DTerrainCfg: 勾配 tanθ。x 方向だけの谷（up）／尾根（down）。前後脚の荷重配分を見る
- HfStep1DTerrainCfg: 段差の高さ。中央から距離 d に 1 段（up/down）。脚長・腰高を見る
- HfCorrugationTerrainCfg: 波長（振幅は一定）。x 方向の波板。胴長・背骨関節を見る
- HfCrossSlopeTerrainCfg: 横傾斜 tanθ。y 方向に傾いた帯を x 方向に歩く。胴幅・歩隔を見る
- HfRoughFieldTerrainCfg: 凹凸の RMS 高さ または 相関長。ガウス／フォン・カルマン型の乱数場。不整地の頑健性を見る
- MeshGap1DTerrainCfg: 溝の幅。x 方向に一定間隔の溝。空中期・跳躍を見る
- MeshBeam1DTerrainCfg: 足場の幅。中央の台から +x/−x に伸びる 1 本の梁。左右バランスを見る

共通の決まり
- snap_levels を行数（num_rows）にすると、行 i の難易度が i/(num_rows−1) ちょうどに丸められる
  （生成器は同じ行の中で難易度に乱数を足すので、評価ではこれで値を固定する）。
  各行の実際の値は level_values(cfg, num_rows) で取り出せる。
- 高さマップ型（Hf…）は Newton の高さマップ当たり判定に変換できる。メッシュ型（Mesh…）を
  1 つでも混ぜると全体がメッシュの当たり判定になる（生成器の仕様）。
- 1 方向だけに変化する地形なので、y 方向の端は隣のマスとの間に崖ができる。
  学習でランダムな向きを指令する場合は既存の角錐型（HfPyramidSloped… など）を使い、これらは評価用にする。
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import Literal

import numpy as np
import torch
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

# ===========================================================================
# 共通の部品
# ===========================================================================


def _level(difficulty: float, cfg) -> float:
    """snap_levels があれば難易度を i/(n−1) に丸める（行 i の難易度は [i/n, (i+1)/n) に入るため）。"""
    n = getattr(cfg, "snap_levels", None)
    d = float(np.clip(difficulty, 0.0, 1.0))
    if not n or n < 2:
        return d
    i = min(int(math.floor(d * n + 1e-9)), n - 1)
    return i / (n - 1)


def _lerp(rng: tuple[float, float], d: float, log: bool = False) -> float:
    a, b = rng
    if log:
        return float(math.exp(math.log(a) + d * (math.log(b) - math.log(a))))
    return float(a + d * (b - a))


def height_field_to_mesh_local(window: float = 0.3):
    """isaaclab の height_field_to_mesh と同じ変換をし、原点の高さだけを
    「中心 ±window [m] の最高点」にする版（既定の ±1 m だと傾斜地でロボットが浮いて出現するため）。

    内側の関数は (difficulty, cfg, xs, ys) を受け取り、高さ [m] の float 配列（x, y の順）を返せばよい。
    """

    def deco(func):
        @functools.wraps(func)
        def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
            if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
                raise ValueError("border_width must be >= horizontal_scale")
            hs, vs = cfg.horizontal_scale, cfg.vertical_scale
            W = int(round(cfg.size[0] / hs)) + 1
            L = int(round(cfg.size[1] / hs)) + 1
            b = int(cfg.border_width / hs) + 1
            nx, ny = W - 2 * b, L - 2 * b
            # 中心が 0 の座標（外周 1 マス以上は高さ 0 の余白）
            xs = (np.arange(nx) - (nx - 1) / 2) * hs
            ys = (np.arange(ny) - (ny - 1) / 2) * hs
            h_m = np.asarray(func(difficulty, cfg, xs, ys), dtype=np.float64)
            assert h_m.shape == (nx, ny), (h_m.shape, (nx, ny))
            heights = np.zeros((W, L), dtype=np.int16)
            heights[b:-b, b:-b] = np.rint(h_m / vs).astype(np.int16)
            vertices, triangles = convert_height_field_to_mesh(heights, hs, vs, cfg.slope_threshold)
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            cx, cy = (W - 1) // 2, (L - 1) // 2
            r = max(int(round(window / hs)), 0)
            origin_z = heights[cx - r : cx + r + 1, cy - r : cy + r + 1].max() * vs
            origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
            return [mesh], origin

        return wrapper

    return deco


@configclass
class _SweepHfCfg(HfTerrainBaseCfg):
    snap_levels: int | None = None
    """評価用：行数と同じ値にすると、行ごとの値が固定される"""


@configclass
class _SweepMeshCfg(SubTerrainBaseCfg):
    snap_levels: int | None = None


# ===========================================================================
# 1. 1 次元の斜面（谷 = 上り、尾根 = 下り）
# ===========================================================================


def slope1d_param(d: float, cfg: HfSlope1DTerrainCfg) -> float:
    return _lerp(cfg.slope_range, d)


@height_field_to_mesh_local()
def slope1d_terrain(difficulty, cfg: HfSlope1DTerrainCfg, xs, ys):
    d = _level(difficulty, cfg)
    t = slope1d_param(d, cfg)
    half = xs.max()
    ramp = max(half - cfg.platform_width / 2 - cfg.edge_flat, 0.0)
    u = np.abs(xs)
    # 中心からの距離 → 斜面上の位置（0：台の縁、ramp：外側の平地）
    s = np.clip(u - cfg.platform_width / 2, 0.0, ramp)
    prof = t * (s - ramp)  # 谷：中心が −t·ramp、外側で 0（中心から外へ上り）
    if cfg.direction == "down":
        prof = -prof  # 尾根：中心が +t·ramp、外側で 0（中心から外へ下り）
    return np.repeat(prof[:, None], ys.size, axis=1)


@configclass
class HfSlope1DTerrainCfg(_SweepHfCfg):
    function = slope1d_terrain
    direction: Literal["up", "down"] = "up"
    slope_range: tuple[float, float] = (0.0, 0.4)
    """勾配 tanθ（0.4 ≈ 21.8°）"""
    platform_width: float = 1.0
    edge_flat: float = 0.5
    """x 端の平地の長さ（m）。斜面の長さ = size/2 − platform/2 − edge_flat"""


# ===========================================================================
# 2. 1 段の段差（上り：中心が低い／下り：中心が高い）
# ===========================================================================


def step1d_param(d: float, cfg: HfStep1DTerrainCfg) -> float:
    return _lerp(cfg.step_height_range, d)


@height_field_to_mesh_local()
def step1d_terrain(difficulty, cfg: HfStep1DTerrainCfg, xs, ys):
    d = _level(difficulty, cfg)
    H = step1d_param(d, cfg)
    inner = np.abs(xs) < cfg.step_distance
    prof = np.where(inner, -H if cfg.direction == "up" else H, 0.0)
    return np.repeat(prof[:, None], ys.size, axis=1)


@configclass
class HfStep1DTerrainCfg(_SweepHfCfg):
    function = step1d_terrain
    direction: Literal["up", "down"] = "up"
    step_height_range: tuple[float, float] = (0.02, 0.30)
    step_distance: float = 1.0
    """中心から段差の面までの距離（m）。horizontal_scale 単位に丸められる"""
    # 段差の面を垂直にするには slope_threshold（0.75 など）が必要。TerrainGenerator は
    # horizontal_scale・vertical_scale・slope_threshold を生成器側の値で上書きするので、
    # 生成器の slope_threshold（既定 0.75）を None にしないこと。
    # Newton の高さマップ当たり判定に変換すると段差は 1 マス（0.1 m）の急斜面になる。
    # 厳密な垂直面が必要なら、その地形セットでは convert_to_heightfield=False にする。


# ===========================================================================
# 3. 波板（波長を変え、振幅は一定）
# ===========================================================================


def corrugation_param(d: float, cfg: HfCorrugationTerrainCfg) -> float:
    return _lerp(cfg.wavelength_range, d, log=cfg.log_spacing)


@height_field_to_mesh_local()
def corrugation_terrain(difficulty, cfg: HfCorrugationTerrainCfg, xs, ys):
    d = _level(difficulty, cfg)
    lam = corrugation_param(d, cfg)
    if lam < 4 * cfg.horizontal_scale:
        raise ValueError(f"wavelength {lam:.3f} m < 4 × horizontal_scale; 波形が表現できない")
    half = xs.max()
    avail = half - cfg.platform_width / 2 - cfg.edge_flat
    n = int(math.floor(avail / lam + 1e-9))  # 端で高さ 0 に戻るよう整数個だけ
    if n < cfg.min_waves:
        raise ValueError(f"wavelength {lam:.2f} m: 波が {n} 個しか入らない。size を大きくするか範囲を狭める")
    s = np.abs(xs) - cfg.platform_width / 2
    on = (s >= 0) & (s < n * lam)
    prof = np.where(on, 0.5 * cfg.amplitude * (1.0 - np.cos(2 * np.pi * s / lam)), 0.0)
    return np.repeat(prof[:, None], ys.size, axis=1)


@configclass
class HfCorrugationTerrainCfg(_SweepHfCfg):
    function = corrugation_terrain
    wavelength_range: tuple[float, float] = (0.4, 2.0)
    """波長（m）。胴の長さ（前後の肩・股関節の距離）の 1〜3 倍程度を含めるとよい"""
    log_spacing: bool = True
    """波長を対数で等間隔にする"""
    amplitude: float = 0.08
    """山と谷の高低差（m）"""
    platform_width: float = 1.0
    edge_flat: float = 0.3
    min_waves: int = 2


# ===========================================================================
# 4. 横傾斜（y 方向に傾いた帯を +x 方向に歩く）
# ===========================================================================


def cross_slope_param(d: float, cfg: HfCrossSlopeTerrainCfg) -> float:
    return _lerp(cfg.slope_range, d)


@height_field_to_mesh_local()
def cross_slope_terrain(difficulty, cfg: HfCrossSlopeTerrainCfg, xs, ys):
    d = _level(difficulty, cfg)
    t = cross_slope_param(d, cfg)
    half_y = ys.max()
    band = half_y / 2  # |y| < band が一定の横傾斜。外側は折り返して y 端で 0 に戻る
    s = np.where(np.abs(ys) <= band, ys, np.sign(ys) * (2 * band - np.abs(ys)))
    prof_y = t * s
    # x 端では中心線 y=0 だけが高さ 0 で隣のマスとつながる（ロボットは中心線上を歩く想定）
    return np.repeat(prof_y[None, :], xs.size, axis=0)


@configclass
class HfCrossSlopeTerrainCfg(_SweepHfCfg):
    function = cross_slope_terrain
    slope_range: tuple[float, float] = (0.0, 0.4)
    """横傾斜 tanθ。一定の傾斜の帯の幅は size[1]/2"""


# ===========================================================================
# 5. 乱数場の凹凸（RMS 高さ と 相関長 を別々に変えられる）
# ===========================================================================


def rough_param(d: float, cfg: HfRoughFieldTerrainCfg) -> float:
    if cfg.sweep == "rms":
        return _lerp(cfg.rms_range, d)
    return _lerp(cfg.corr_length_range, d, log=True)


def _cos_taper(dist: np.ndarray, start: float, width: float) -> np.ndarray:
    """dist ≤ start で 0、start+width 以上で 1 の滑らかな窓"""
    if width <= 0:
        return (dist > start).astype(float)
    z = np.clip((dist - start) / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * z)


@height_field_to_mesh_local()
def rough_field_terrain(difficulty, cfg: HfRoughFieldTerrainCfg, xs, ys):
    d = _level(difficulty, cfg)
    if cfg.sweep == "rms":
        rms, ell = rough_param(d, cfg), cfg.corr_length
    else:
        rms, ell = cfg.rms, rough_param(d, cfg)
    nx, ny, hs = xs.size, ys.size, cfg.horizontal_scale
    # 乱数：seed を決めると、行（難易度）ごとに同じ地形になる → 全形態で同じ地面を使える
    if cfg.seed is None:
        noise = np.random.normal(size=(nx, ny))
    else:
        level_id = int(round(d * 10_000))
        noise = np.random.default_rng([cfg.seed, level_id]).normal(size=(nx, ny))
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=hs)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=hs)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    if cfg.spectrum == "gaussian":
        amp = np.exp(-k2 * ell**2 / 8.0)  # 自己相関 exp(−r²/ℓ²)
    else:  # フォン・カルマン（自己相似な凹凸。hurst が小さいほど細かい凹凸が多い）
        amp = (1.0 + k2 * ell**2) ** (-(cfg.hurst + 1.0) / 2.0)
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * amp))
    field -= field.mean()
    field *= rms / max(field.std(), 1e-12)
    # 中央の台（高さ 0）と外周（高さ 0）へ滑らかにつなぐ
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    w_center = _cos_taper(np.maximum(np.abs(X), np.abs(Y)), cfg.platform_width / 2, cfg.blend_width)
    edge = np.minimum(xs.max() - np.abs(X), ys.max() - np.abs(Y))
    w_edge = _cos_taper(edge, 0.0, cfg.blend_width)
    return field * w_center * w_edge


@configclass
class HfRoughFieldTerrainCfg(_SweepHfCfg):
    function = rough_field_terrain
    sweep: Literal["rms", "corr"] = "rms"
    """難易度で動かす量：rms（凹凸の大きさ）または corr（凹凸の細かさ）"""
    rms_range: tuple[float, float] = (0.0, 0.06)
    corr_length: float = 0.3
    """sweep="rms" のときの相関長（m）"""
    rms: float = 0.03
    """sweep="corr" のときの RMS 高さ（m）"""
    corr_length_range: tuple[float, float] = (0.15, 1.2)
    """sweep="corr" のときの相関長の範囲（m、対数で等間隔）。下限は 1.5 × horizontal_scale 以上に"""
    # 注意：生成器の slope_threshold（既定 0.75）より急な所は垂直面に直される。
    # rms 0.06 m・相関長 0.3 m で全体の約 0.6%。気になるなら相関長を長めにする。
    spectrum: Literal["gaussian", "vonkarman"] = "gaussian"
    hurst: float = 0.8
    platform_width: float = 1.0
    blend_width: float = 0.5
    seed: int | None = 0
    """None なら生成器の乱数（環境の seed 依存）。評価では固定値にする"""


# ===========================================================================
# 6. 溝（メッシュ）
# ===========================================================================


def gap1d_param(d: float, cfg: MeshGap1DTerrainCfg) -> float:
    return _lerp(cfg.gap_width_range, d)


def gap1d_terrain(difficulty, cfg: MeshGap1DTerrainCfg):
    d = _level(difficulty, cfg)
    w = gap1d_param(d, cfg)
    sx, sy = cfg.size
    half = sx / 2
    # 中心から外へ：台 → (溝 w → 足場 solid_length) の繰り返し。端に溝が入りきらなければ端まで足場
    solids = [(-cfg.platform_width / 2, cfg.platform_width / 2)]
    for sign in (+1, -1):
        x = cfg.platform_width / 2
        while x < half - 1e-6:
            if x + w + cfg.min_solid_end > half:
                solids.append(tuple(sorted((sign * x, sign * half))))
                break
            x_end = min(x + w + cfg.solid_length, half)
            solids.append(tuple(sorted((sign * (x + w), sign * x_end))))
            x = x_end
    meshes = []
    for a, b in solids:
        box = trimesh.creation.box((b - a, sy, cfg.gap_depth))
        box.apply_translation(((a + b) / 2 + half, sy / 2, -cfg.gap_depth / 2))
        meshes.append(box)
    floor = trimesh.creation.box((sx, sy, 0.05))
    floor.apply_translation((half, sy / 2, -cfg.gap_depth - 0.025))
    meshes.append(floor)
    return meshes, np.array([half, sy / 2, 0.0])


@configclass
class MeshGap1DTerrainCfg(_SweepMeshCfg):
    function = gap1d_terrain
    gap_width_range: tuple[float, float] = (0.05, 0.40)
    solid_length: float = 1.0
    """溝と溝の間の足場の長さ（m）"""
    gap_depth: float = 0.5
    platform_width: float = 1.0
    min_solid_end: float = 0.3
    """端の足場がこれより短くなる溝は作らない（m）"""


# ===========================================================================
# 7. 細い梁（メッシュ）
# ===========================================================================


def beam1d_param(d: float, cfg: MeshBeam1DTerrainCfg) -> float:
    # 難易度が上がるほど細くなる
    return _lerp(cfg.beam_width_range, d)


def beam1d_terrain(difficulty, cfg: MeshBeam1DTerrainCfg):
    d = _level(difficulty, cfg)
    w = beam1d_param(d, cfg)
    sx, sy = cfg.size
    D = cfg.trench_depth
    plat = trimesh.creation.box((cfg.platform_width, cfg.platform_width, D))
    plat.apply_translation((sx / 2, sy / 2, -D / 2))
    beam = trimesh.creation.box((sx, w, D))
    beam.apply_translation((sx / 2, sy / 2, -D / 2))
    floor = trimesh.creation.box((sx, sy, 0.05))
    floor.apply_translation((sx / 2, sy / 2, -D - 0.025))
    return [plat, beam, floor], np.array([sx / 2, sy / 2, 0.0])


@configclass
class MeshBeam1DTerrainCfg(_SweepMeshCfg):
    function = beam1d_terrain
    beam_width_range: tuple[float, float] = (0.5, 0.12)
    """梁の幅（m）。左が難易度 0"""
    trench_depth: float = 0.3
    platform_width: float = 1.0


# ===========================================================================
# 行ごとの値の取り出し（解析・表作成用）
# ===========================================================================

_PARAM = {
    HfSlope1DTerrainCfg: ("slope_tan", slope1d_param),
    HfStep1DTerrainCfg: ("step_height_m", step1d_param),
    HfCorrugationTerrainCfg: ("wavelength_m", corrugation_param),
    HfCrossSlopeTerrainCfg: ("cross_slope_tan", cross_slope_param),
    HfRoughFieldTerrainCfg: (None, rough_param),
    MeshGap1DTerrainCfg: ("gap_width_m", gap1d_param),
    MeshBeam1DTerrainCfg: ("beam_width_m", beam1d_param),
}


def level_values(cfg, num_rows: int) -> tuple[str, list[float]]:
    """snap_levels=num_rows のとき、行 0..num_rows−1 の実際の値を返す。"""
    name, fn = _PARAM[type(cfg)]
    if name is None:
        name = "rms_m" if cfg.sweep == "rms" else "corr_length_m"
    vals = [fn(i / (num_rows - 1), cfg) for i in range(num_rows)]
    if isinstance(cfg, HfStep1DTerrainCfg):  # 段差の位置は 0.1 m 単位だが高さは vertical_scale 単位
        vals = [round(v / cfg.vertical_scale) * cfg.vertical_scale for v in vals]
    return name, vals


# ===========================================================================
# 地形セット
# ===========================================================================


def make_eval6_terrain_cfg(num_rows: int = 10, size: float = 10.0, extras: bool = False, seed: int = 12345):
    """列＝地形の種類、行＝難易度（値は level_values で確認）。curriculum=True で行が難易度順に並ぶ。

    extras=True で凹凸（RMS）・溝・梁を追加する（メッシュ型を含むので高さマップ変換は無効になる）。
    """
    from isaaclab.terrains import TerrainGeneratorCfg

    kw = {"proportion": 1.0, "snap_levels": num_rows}
    subs = {
        # 平地も高さマップ型にしておく（MeshPlane はメッシュ型なので高さマップ変換が無効になる）
        "flat": HfSlope1DTerrainCfg(slope_range=(0.0, 0.0), **kw),
        "slope_up": HfSlope1DTerrainCfg(direction="up", **kw),
        "slope_down": HfSlope1DTerrainCfg(direction="down", **kw),
        "step_up": HfStep1DTerrainCfg(direction="up", **kw),
        "step_down": HfStep1DTerrainCfg(direction="down", **kw),
        "corrugation": HfCorrugationTerrainCfg(**kw),
        "cross_slope": HfCrossSlopeTerrainCfg(**kw),
    }
    if extras:
        subs |= {
            "rough_rms": HfRoughFieldTerrainCfg(sweep="rms", **kw),
            "gap": MeshGap1DTerrainCfg(**kw),
            "beam": MeshBeam1DTerrainCfg(**kw),
        }
    return TerrainGeneratorCfg(
        size=(size, size),
        border_width=5.0,
        num_rows=num_rows,
        num_cols=len(subs),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        seed=seed,
        sub_terrains=subs,
    )


def pin_envs_to_cells(env) -> tuple[torch.Tensor, torch.Tensor]:
    """env を (難易度 level, 地形種 type) の全セルへラウンドロビンで均等配置する。

    gym.make() の後・評価ループの前に呼び、その後 env.reset() でスポーン位置を反映させる。
    num_envs はセル数（行数 × 列数）の倍数にすると各セルの試行数が揃う。
    戻り値の levels / types を各 env の指標と一緒に保存しておけば、地形別・難易度別に層別集計できる。
    """
    uenv = env.unwrapped
    terrain = uenv.scene.terrain
    if terrain.terrain_origins is None:
        raise RuntimeError("terrain_type='generator' かつ use_terrain_origins=True のときだけ使える")
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    n_cells = num_rows * num_cols
    n = uenv.num_envs
    if n % n_cells != 0:
        warnings.warn(f"num_envs={n} がセル数 {n_cells} の倍数でない → セルごとの試行数が不揃いになる", stacklevel=2)
    cell = torch.arange(n, device=terrain.terrain_origins.device) % n_cells
    terrain.terrain_levels[:] = cell // num_cols
    terrain.terrain_types[:] = cell % num_cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    return terrain.terrain_levels.clone(), terrain.terrain_types.clone()
