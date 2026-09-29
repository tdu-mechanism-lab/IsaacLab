# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL.

定量評価用ログ機能を追加した版:
  * 時系列ログ  : logs/.../eval_<tag>_timeseries.csv  (指定した1環境のみ)
  * 集計指標    : logs/.../eval_<tag>_summary.csv / .json  (全環境で集計)

集計される指標:
  - Cost of Transport (CoT)      : 無次元. E / (m g d)
  - 正規化トルク  tau/(m g l)    : ピーク / RMS
  - トルク飽和率                 : effort limit に対する張り付き時間割合
  - 前脚 / 後脚 の負荷分担比
  - Duty factor, ストライド周波数, 無次元ストライド長
  - 速度追従 RMSE, Froude 数
  - 体幹 roll / pitch の RMS, 胴体高さ変動
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ----------------------------- 評価ログ用の引数 -----------------------------
parser.add_argument("--eval_steps", type=int, default=1000, help="評価に使うステップ数 (warmup を除く). 0 で無制限.")
parser.add_argument("--warmup_steps", type=int, default=100, help="集計から除外する初期ステップ数 (姿勢が落ち着くまで).")
parser.add_argument("--log_env", type=int, default=0, help="時系列 CSV を出力する環境インデックス.")
parser.add_argument("--eval_tag", type=str, default="run", help="出力ファイル名につけるタグ (例: dog, horse).")
parser.add_argument("--leg_length", type=float, default=None, help="正規化に使う脚長 [m]. 未指定なら立位の股関節高さを使用.")
parser.add_argument("--contact_threshold", type=float, default=5.0,
                    help="接地判定の力しきい値 [N]. 体重の数%程度が目安.")
parser.add_argument("--live_duty_threshold", type=float, default=0.05,
                    help="この duty 未満の脚は「機能していない」とみなし, duty/浮遊/歩容の集計から除外する.")
parser.add_argument("--drag_pattern", type=str, default=".*_(thigh|calf|calf2)$",
                    help="足先以外の接地(引きずり)を検出するボディ名の正規表現.")
parser.add_argument("--contact_min_steps", type=int, default=2,
                    help="これ未満の長さの接地/浮遊区間を除去する (チャタリング対策). 1 で無効. "
                         "接地時間が短い走行歩容では 2 を超えると本物の立脚まで消えるので注意.")
parser.add_argument("--foot_pattern", type=str, default=".*_foot", help="足先ボディ名の正規表現.")

# ----------------------------- 追従カメラ用の引数 -----------------------------
parser.add_argument("--follow_cam", action="store_true", default=False, help="ロボットを追従するカメラを有効化.")
parser.add_argument("--cam_distance", type=float, default=2.0, help="カメラの水平距離 [m].")
parser.add_argument("--cam_height", type=float, default=0.4, help="ロボット原点からのカメラ高さ [m].")
parser.add_argument("--cam_azimuth", type=float, default=90.0,
                    help="方位角 [deg]. 0=真後ろ, 90=右側面, 180=正面, 270=左側面.")
parser.add_argument("--cam_lookat_z", type=float, default=0.0, help="注視点の高さオフセット [m].")
parser.add_argument("--cam_follow_heading", action="store_true", default=False,
                    help="機体のヨー角にも追従する (旋回する場合に使用).")
parser.add_argument("--cam_smooth", type=float, default=0.85,
                    help="カメラ位置の平滑化係数. 0=即応, 1に近いほど滑らか.")
parser.add_argument("--cam_env", type=int, default=None, help="追従する環境インデックス. 未指定なら --log_env と同じ.")
parser.add_argument("--cam_resolution", type=int, nargs=2, default=[1920, 1080], help="録画解像度 (幅 高さ).")
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import math
import os
import time
import csv
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

GRAVITY = 9.81


# ============================================================================
# ヘルパ
# ============================================================================
def _find_contact_sensor(scene):
    """シーンから接触センサを名前で探す (見つからなければ None)."""
    for key in scene.sensors.keys():
        if "contact" in key.lower():
            return scene.sensors[key]
    return None


def _get_actuator_params(robot, num_envs, num_joints, device):
    """アクチュエータの effort_limit / saturation_effort / velocity_limit を (N, J) で返す.

    DCMotor はトルク速度曲線を持つため, 実効上限は角速度に依存する.
    saturation_effort が取得できない場合 (ImplicitActuator 等) は NaN を返し,
    呼び出し側は定数の effort_limit にフォールバックする.
    """
    out = {}
    for key in ("effort_limit", "saturation_effort", "velocity_limit"):
        out[key] = torch.full((num_envs, num_joints), float("nan"), device=device)
    for name, act in getattr(robot, "actuators", {}).items():
        idx = act.joint_indices
        for key in out:
            v = getattr(act, key, None)
            if v is None:
                continue
            try:
                t = torch.as_tensor(v, dtype=torch.float32, device=device)
                if t.ndim == 0:
                    out[key][:, idx] = float(t)
                else:
                    out[key][:, idx] = t.reshape(num_envs, -1).to(device)
            except Exception:
                pass
        print(f"[INFO] Actuator '{name}': "
              + ", ".join(f"{k}={float(torch.as_tensor(getattr(act, k)).flatten()[0]):.3f}"
                          for k in out if getattr(act, k, None) is not None))
    return out["effort_limit"], out["saturation_effort"], out["velocity_limit"]


def _effective_torque_limit(tau, qd, eff, sat, vel):
    """その瞬間に出せるトルクの上限 (符号ごと) を返す.

    DCMotor: max = clip(sat*(1 - qd/vel), 0, eff),  min = clip(sat*(-1 - qd/vel), -eff, 0)
    定数 effort_limit だけで判定すると, 高速回転時の飽和を取りこぼす.
    """
    if torch.isnan(sat).any() or torch.isnan(vel).any():
        return eff
    ratio = qd / torch.clamp(vel, min=1e-6)
    max_e = torch.minimum(torch.clamp(sat * (1.0 - ratio), min=0.0), eff)
    min_e = torch.maximum(torch.clamp(sat * (-1.0 - ratio), max=0.0), -eff)
    return torch.where(tau >= 0, max_e, -min_e)


def _get_effort_limits(robot, num_envs, num_joints, device):
    """関節ごとの実効 effort limit を (num_envs, num_joints) で返す.

    robot.data.joint_effort_limits はシミュレータ側の上限 (しばしば 1e9) なので,
    実際にトルクを制限しているアクチュエータモデルの effort_limit を優先して読む.
    """
    lim = torch.full((num_envs, num_joints), float("nan"), device=device)
    found = False
    for name, act in getattr(robot, "actuators", {}).items():
        el = getattr(act, "effort_limit", None)
        if el is None:
            continue
        idx = act.joint_indices
        try:
            el_t = torch.as_tensor(el, dtype=torch.float32, device=device)
            if el_t.ndim == 0:
                lim[:, idx] = float(el_t)
            else:
                lim[:, idx] = el_t.reshape(num_envs, -1).to(device)
            found = True
            print(f"[INFO] Actuator '{name}': effort_limit = {float(torch.as_tensor(el).flatten()[0]):.3f} N*m")
        except Exception as exc:
            print(f"[WARN] アクチュエータ '{name}' の effort_limit を解釈できません: {exc}")

    if not found:
        for attr in ("joint_effort_limits", "joint_effort_limit", "joint_effort_limit_sim"):
            val = getattr(robot.data, attr, None)
            if val is not None:
                lim = val.clone().to(device).reshape(num_envs, -1).float()
                found = True
                break

    if not found:
        print("[WARN] effort limit を取得できませんでした. 飽和率は NaN になります.")
    elif float(torch.nanmedian(lim)) > 1.0e6:
        print("[WARN] effort limit が異常に大きい値です. 飽和率は意味を持ちません.")
    return lim


def _set_camera(env, eye, target):
    """ビューポートカメラを世界座標で置き直す. API 名がバージョンで違うため順に試す."""
    ctrl = getattr(env.unwrapped, "viewport_camera_controller", None)
    if ctrl is not None:
        try:
            ctrl.update_view_location(eye=list(eye), lookat=list(target))
            return True
        except Exception:
            pass
    try:
        env.unwrapped.sim.set_camera_view(eye=tuple(eye), target=tuple(target))
        return True
    except Exception:
        return False


def _wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _debounce_contacts(contact_bool, min_steps):
    """min_steps 未満の接地/浮遊区間を直前の状態で埋める (チャタリング除去).

    contact_bool: (T, num_envs, num_feet) の bool ndarray.
    力しきい値をノイズで跨ぐ 1-2 ステップの偽イベントを消し,
    duty factor とストライド周波数を実際の歩容に一致させる.
    """
    if min_steps <= 1:
        return contact_bool
    out = contact_bool.copy()
    T = out.shape[0]
    flat = out.reshape(T, -1)
    for k in range(flat.shape[1]):
        s = flat[:, k]
        i = 0
        while i < T:
            j = i
            while j < T and s[j] == s[i]:
                j += 1
            if (j - i) < min_steps and i > 0:
                s[i:j] = s[i - 1]
            i = j
        flat[:, k] = s
    return flat.reshape(out.shape)


def _gait_analysis(contact_b, foot_names, dt, speed_mean, leg_len, live_thr=0.05):
    """環境ごとに歩容を解析する.

    contact_b: (T, N, F) の bool. 戻り値はすべて (N,) の配列.

    ストライド周期は「基準肢の接地開始間隔の中央値」で定義する (1肢1周期に1回接地する
    という歩容の定義そのもの). 周期を自由探索すると 2 倍・1/2 倍のエイリアスに落ちるため
    行わない. 相対位相は基準肢の直前の接地からの経過を周期で割った値で, 円周統計の
    集中度 R と Rayleigh 検定で「一定の歩容に収束しているか」を判定する.
    """
    T, N, F = contact_b.shape
    names = [n.replace("_foot", "") for n in foot_names]
    out = {k: np.full(N, np.nan) for k in
           ["stride_period_s", "stride_frequency_hz", "stride_length_normalized",
            "duty_live", "flight_live", "n_live_legs", "gait_phase_R", "gait_phase_p",
            "hind_phase_offset", "fore_hind_phase_offset"]}
    per_leg_phase = np.full((N, F), np.nan)

    # 基準肢の優先順位: RL -> RR -> FL -> FR (機能している最初のもの)
    pref = [i for nm in ("RL", "RR", "FL", "FR") for i, n in enumerate(names) if n == nm]

    for e in range(N):
        C = contact_b[:, e, :]
        duty = C.mean(axis=0)
        live = np.where(duty > live_thr)[0]
        out["n_live_legs"][e] = len(live)
        if len(live) == 0:
            continue
        out["duty_live"][e] = duty[live].mean()
        out["flight_live"][e] = (C[:, live].sum(axis=1) == 0).mean()
        ref = next((i for i in pref if i in live), live[0])

        tds = {}
        for i in live:
            idx = np.where(np.logical_and(C[1:, i], ~C[:-1, i]))[0] + 1
            tds[i] = idx * dt
        if len(tds[ref]) < 4:
            continue
        iv = np.diff(tds[ref])
        iv = iv[(iv > 3 * dt) & (iv < 0.5 * T * dt)]
        if len(iv) < 3:
            continue
        Ts = float(np.median(iv))
        out["stride_period_s"][e] = Ts
        out["stride_frequency_hz"][e] = 1.0 / Ts
        out["stride_length_normalized"][e] = speed_mean[e] * Ts / max(leg_len, 1e-6)

        Rs, ps = [], []
        for i in live:
            if i == ref:
                per_leg_phase[e, i] = 0.0
                continue
            ph = []
            for t in tds[i]:
                prev = tds[ref][tds[ref] <= t]
                if len(prev) == 0:
                    continue
                d = (t - prev[-1]) / Ts
                if d < 1.2:
                    ph.append(d % 1.0)
            if len(ph) < 5:
                continue
            ang = 2 * np.pi * np.asarray(ph)
            z = np.mean(np.exp(1j * ang))
            R = float(np.abs(z))
            mu = float((np.angle(z) / (2 * np.pi)) % 1.0)
            n = len(ph)
            Z = n * R * R
            pval = float(np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n)))
            per_leg_phase[e, i] = mu
            Rs.append(R)
            ps.append(min(1.0, pval))
        if Rs:
            out["gait_phase_R"][e] = float(np.mean(Rs))
            out["gait_phase_p"][e] = float(np.max(ps))

    # 後肢ペアの位相差, 前肢-後肢の位相差 (0.5 に折り返した値)
    def _fold(x):
        return np.where(np.isnan(x), np.nan, np.minimum(x % 1.0, (1.0 - x) % 1.0))
    idx = {n: i for i, n in enumerate(names)}
    if "RL" in idx and "RR" in idx:
        out["hind_phase_offset"] = _fold(per_leg_phase[:, idx["RR"]] - per_leg_phase[:, idx["RL"]])
    fore = [idx[n] for n in ("FL", "FR") if n in idx]
    hind = [idx[n] for n in ("RL", "RR") if n in idx]
    if fore and hind:
        with np.errstate(invalid="ignore"):
            fm = np.nanmean(per_leg_phase[:, fore], axis=1)
            hm = np.nanmean(per_leg_phase[:, hind], axis=1)
        out["fore_hind_phase_offset"] = _fold(fm - hm)
    return out, per_leg_phase


def _count_contact_onsets(contact_bool):
    """接地の立ち上がり回数を数える. contact_bool: (T, num_envs, num_feet) の bool ndarray."""
    if contact_bool.shape[0] < 2:
        return np.zeros(contact_bool.shape[1:], dtype=np.int64)
    onset = np.logical_and(contact_bool[1:], np.logical_not(contact_bool[:-1]))
    return onset.sum(axis=0)


# ============================================================================
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # ------------------------------------------------------------------
    # 追従カメラの設定 (gym.make より前に行う必要がある)
    # ------------------------------------------------------------------
    cam_env_idx = args_cli.cam_env if args_cli.cam_env is not None else args_cli.log_env
    if args_cli.follow_cam:
        az = math.radians(args_cli.cam_azimuth)
        d = args_cli.cam_distance
        # ロボット基準の水平オフセット: az=0 で真後ろ, 90 で右側面
        off_x = -d * math.cos(az)
        off_y = -d * math.sin(az)

        env_cfg.viewer.resolution = tuple(args_cli.cam_resolution)
        env_cfg.viewer.env_index = cam_env_idx

        if args_cli.cam_follow_heading:
            # ヨー角にも追従させる場合は毎ステップ手動で置き直すので world のまま
            env_cfg.viewer.origin_type = "world"
        else:
            # IsaacLab 組み込みの asset_root 追従 (ワールド軸に固定したオフセット)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.eye = (off_x, off_y, args_cli.cam_height)
            env_cfg.viewer.lookat = (0.0, 0.0, args_cli.cam_lookat_z)
        print(f"[INFO] Follow camera: env={cam_env_idx}, offset=({off_x:.2f}, {off_y:.2f}, "
              f"{args_cli.cam_height:.2f}), follow_heading={args_cli.cam_follow_heading}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    # ========================================================================
    # 評価用のセットアップ
    # ========================================================================
    scene = env.unwrapped.scene
    robot = scene["robot"]
    joint_names = list(robot.data.joint_names)
    num_joints = len(joint_names)
    num_envs = env.unwrapped.num_envs

    # --- 足先ボディの特定 ---
    foot_ids, foot_names = robot.find_bodies(args_cli.foot_pattern)
    if len(foot_ids) == 0:
        print(f"[WARN] '{args_cli.foot_pattern}' に一致するボディがありません. 接地指標は無効になります.")
    else:
        print(f"[INFO] Foot bodies: {foot_names}")

    # --- 接触センサ ---
    contact_sensor = _find_contact_sensor(scene)
    sensor_foot_ids = []
    sensor_foot_names = []
    if contact_sensor is not None and len(foot_ids) > 0:
        sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(args_cli.foot_pattern)
        print(f"[INFO] Contact sensor foot bodies: {sensor_foot_names}")
    # 床反力の前後分担を出すため, センサ側の足の並びで前脚/後脚を判別する
    sf_front = [i for i, n in enumerate(sensor_foot_names) if n.startswith("F")]
    sf_rear = [i for i, n in enumerate(sensor_foot_names) if n.startswith("R")]
    if sensor_foot_names:
        print(f"[INFO] GRF 前脚={[sensor_foot_names[i] for i in sf_front]} "
              f"後脚={[sensor_foot_names[i] for i in sf_rear]}")
    drag_ids = []
    if contact_sensor is not None:
        try:
            drag_ids, drag_names = contact_sensor.find_bodies(args_cli.drag_pattern)
            print(f"[INFO] 引きずり検出対象 ({len(drag_ids)}): {drag_names}")
        except Exception as exc:
            print(f"[WARN] 引きずり検出対象を解決できません ({exc}). 該当指標は NaN になります.")
    if contact_sensor is None:
        print("[WARN] 接触センサが見つかりません. duty factor / ストライド周波数は NaN になります.")

    # --- 前脚 / 後脚の関節インデックス (URDF の F*/R* 命名を前提) ---
    front_j = [i for i, n in enumerate(joint_names) if n.startswith("F")]
    rear_j = [i for i, n in enumerate(joint_names) if n.startswith("R")]
    print(f"[INFO] Front joints ({len(front_j)}): {[joint_names[i] for i in front_j]}")
    print(f"[INFO] Rear  joints ({len(rear_j)}): {[joint_names[i] for i in rear_j]}")

    # --- 質量 (正規化用). add_base_mass 等で環境ごとに異なりうるので個別に取得 ---
    try:
        masses = torch.as_tensor(robot.root_physx_view.get_masses()).sum(dim=1).cpu().numpy().astype(np.float64)
    except Exception:
        masses = torch.as_tensor(robot.data.default_mass).sum(dim=1).cpu().numpy().astype(np.float64)
    masses = np.broadcast_to(masses.reshape(-1), (num_envs,)).copy()
    total_mass = float(masses.mean())

    effort_limits = _get_effort_limits(robot, num_envs, num_joints, device)
    eff_lim, sat_eff, vel_lim = _get_actuator_params(robot, num_envs, num_joints, device)
    if torch.isnan(eff_lim).all():
        eff_lim = effort_limits

    print(f"[INFO] Total mass = {total_mass:.4f} kg "
          f"(min {masses.min():.4f} / max {masses.max():.4f} / std {masses.std():.4f})")
    if masses.std() > 1e-6:
        print("[WARN] 環境ごとに質量が異なります (add_base_mass 等のランダム化). "
              "CoT は環境ごとの実質量で計算しますが, モデル間比較では無効化を推奨します.")

    # ======================================================================
    # 物理レート積算
    # ----------------------------------------------------------------------
    # env.step() は decimation 回だけ物理ステップを進め, その各回で
    # write_data_to_sim() がアクチュエータを再評価する. つまり tau も qd も
    # physics_dt (既定 0.005 s = 200 Hz) で変化しており, 制御周期 (50 Hz) の
    # スナップショットは 4 回に 1 回しか見ていない.
    # 物理コールバックを登録して 200 Hz で仕事率とトルクを積算する.
    # ======================================================================
    try:
        phys_dt = float(env.unwrapped.physics_dt)
    except Exception:
        phys_dt = float(env.unwrapped.sim.get_physics_dt())
    decim = max(1, int(round(dt / phys_dt)))
    print(f"[INFO] physics_dt={phys_dt:.4f}s ({1/phys_dt:.0f} Hz), "
          f"step_dt={dt:.4f}s ({1/dt:.0f} Hz), decimation={decim}")

    ACC = {
        "active": False,
        "n": 0,
        "e_pos": torch.zeros(num_envs, device=device, dtype=torch.float64),
        "e_abs": torch.zeros(num_envs, device=device, dtype=torch.float64),
        "tau_sq": torch.zeros(num_envs, num_joints, device=device, dtype=torch.float64),
        "tau_peak": torch.zeros(num_envs, num_joints, device=device),
        "sat_const": torch.zeros(num_envs, num_joints, device=device, dtype=torch.float64),
        "sat_curve": torch.zeros(num_envs, num_joints, device=device, dtype=torch.float64),
        "e_front": torch.zeros(num_envs, device=device, dtype=torch.float64),
        "e_rear": torch.zeros(num_envs, device=device, dtype=torch.float64),
    }
    f_idx = torch.tensor(front_j, device=device, dtype=torch.long) if front_j else None
    r_idx = torch.tensor(rear_j, device=device, dtype=torch.long) if rear_j else None

    def _physics_accum(step_size):
        """物理ステップごとに呼ばれる. 遅延バッファを避けて physx から直接読む."""
        if not ACC["active"]:
            return
        try:
            with torch.inference_mode():
                tau = robot.data.applied_torque
                try:
                    qd_p = robot.root_physx_view.get_dof_velocities()
                except Exception:
                    qd_p = robot.data.joint_vel
                p = tau * qd_p
                ACC["e_pos"] += torch.clamp(p, min=0.0).sum(dim=1).double() * phys_dt
                ACC["e_abs"] += p.abs().sum(dim=1).double() * phys_dt
                ACC["tau_sq"] += (tau ** 2).double() * phys_dt
                ACC["tau_peak"] = torch.maximum(ACC["tau_peak"], tau.abs())
                # 飽和判定: 定数 effort_limit と, 速度依存の実効上限の両方
                ACC["sat_const"] += (tau.abs() >= 0.99 * eff_lim).double()
                lim_v = _effective_torque_limit(tau, qd_p, eff_lim, sat_eff, vel_lim)
                ACC["sat_curve"] += ((tau.abs() >= 0.99 * lim_v) & (lim_v > 1e-3)).double()
                if f_idx is not None and r_idx is not None:
                    ACC["e_front"] += torch.clamp(p[:, f_idx], min=0.0).sum(dim=1).double() * phys_dt
                    ACC["e_rear"] += torch.clamp(p[:, r_idx], min=0.0).sum(dim=1).double() * phys_dt
                ACC["n"] += 1
        except Exception as exc:
            if ACC["n"] == 0:
                print(f"[WARN] 物理レート積算に失敗しました: {exc}")
            ACC["active"] = False

    physics_metrics_ok = False
    try:
        env.unwrapped.sim.add_physics_callback("eval_power_accum", _physics_accum)
        physics_metrics_ok = True
        print("[INFO] 物理レート積算のコールバックを登録しました.")
    except Exception as exc:
        print(f"[WARN] add_physics_callback を登録できません ({exc}). 制御周期(50Hz)の値のみ出力します.")

    # --- 速度コマンド名の特定 ---
    command_name = None
    try:
        for name in env.unwrapped.command_manager.active_terms:
            if "vel" in name.lower():
                command_name = name
                break
        if command_name is None and len(env.unwrapped.command_manager.active_terms) > 0:
            command_name = env.unwrapped.command_manager.active_terms[0]
        print(f"[INFO] Velocity command term: {command_name}")
    except Exception:
        print("[WARN] command_manager にアクセスできません. 追従誤差は NaN になります.")

    # --- 出力ファイル ---
    ts_path = os.path.join(log_dir, f"eval_{args_cli.eval_tag}_timeseries.csv")
    csv_file = open(ts_path, "w", newline="")
    writer = csv.writer(csv_file)
    header = ["time_s", "phase"]
    header += ["cmd_vx", "cmd_vy", "cmd_wz"]
    header += ["vx_b", "vy_b", "vz_b", "wx_b", "wy_b", "wz_b"]
    header += ["roll", "pitch", "base_height"]
    header += [f"q_{n}" for n in joint_names]
    header += [f"qd_{n}" for n in joint_names]
    header += [f"tau_{n}" for n in joint_names]
    header += [f"P_{n}" for n in joint_names]
    header += ["P_pos_total", "P_abs_total"]
    header += [f"contact_{n}" for n in foot_names]
    writer.writerow(header)

    # --- 集計用バッファ (全環境, warmup 後のみ蓄積) ---
    buf = {k: [] for k in ["tau", "qd", "p_pos", "p_abs", "speed_xy", "vx_b", "wz_b",
                           "cmd", "roll", "pitch", "base_h", "contact", "grf_f", "grf_r", "drag"]}

    # reset environment
    obs = env.get_observations()
    timestep = 0
    log_env = min(args_cli.log_env, num_envs - 1)
    cam_env = min(cam_env_idx, num_envs - 1)
    cam_state = {"eye": None, "target": None}
    total_steps = args_cli.warmup_steps + args_cli.eval_steps if args_cli.eval_steps > 0 else -1

    print(f"[INFO] Evaluating: warmup={args_cli.warmup_steps}, eval={args_cli.eval_steps}, num_envs={num_envs}")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # ------------------------------------------------------------
            # 状態量の取得
            # ------------------------------------------------------------
            tau = robot.data.applied_torque                      # (N, J)
            qd = robot.data.joint_vel                            # (N, J)
            q = robot.data.joint_pos                             # (N, J)
            p_joint = tau * qd                                   # (N, J) 関節仕事率 [W]
            p_pos = torch.clamp(p_joint, min=0.0).sum(dim=1)     # (N,) 正の仕事率のみ
            p_abs = p_joint.abs().sum(dim=1)                     # (N,) 絶対値

            lin_b = robot.data.root_lin_vel_b                    # (N, 3)
            ang_b = robot.data.root_ang_vel_b                    # (N, 3)
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            roll = torch.as_tensor(roll)
            pitch = torch.as_tensor(pitch)

            # 速度コマンド
            if command_name is not None:
                cmd = env.unwrapped.command_manager.get_command(command_name)[:, :3]
            else:
                cmd = torch.full((num_envs, 3), float("nan"), device=device)

            # 接地判定: 履歴があれば最大値を取り, ステップ間の接地を取りこぼさない
            grf_f = grf_r = None
            if contact_sensor is not None and len(sensor_foot_ids) > 0:
                hist = getattr(contact_sensor.data, "net_forces_w_history", None)
                if hist is not None:
                    fh = hist[:, :, sensor_foot_ids, :]                   # (N, H, F, 3)
                    f_mag = torch.norm(fh, dim=-1).max(dim=1).values      # (N, F)
                    fz = fh[..., 2].clamp(min=0.0).mean(dim=1)            # (N, F) 履歴平均の鉛直成分
                else:
                    nf = contact_sensor.data.net_forces_w[:, sensor_foot_ids, :]
                    f_mag = torch.norm(nf, dim=-1)
                    fz = nf[..., 2].clamp(min=0.0)
                contact = (f_mag > args_cli.contact_threshold).float()
                # 鉛直床反力の力積 [N*s] を前脚/後脚それぞれで積算
                if sf_front and sf_rear:
                    grf_f = fz[:, sf_front].sum(dim=1) * dt
                    grf_r = fz[:, sf_rear].sum(dim=1) * dt
            else:
                contact = torch.full((num_envs, max(len(foot_ids), 1)), float("nan"), device=device)

            # 足先以外(下腿など)の接地 = 引きずりの検出
            if contact_sensor is not None and len(drag_ids) > 0:
                dh = getattr(contact_sensor.data, "net_forces_w_history", None)
                if dh is not None:
                    dmag = torch.norm(dh[:, :, drag_ids, :], dim=-1).max(dim=1).values
                else:
                    dmag = torch.norm(contact_sensor.data.net_forces_w[:, drag_ids, :], dim=-1)
                drag = (dmag > args_cli.contact_threshold).any(dim=1).float()
            else:
                drag = torch.full((num_envs,), float("nan"), device=device)

            # 胴体高さ: 接地している足だけを基準にする.
            # 全足の平均を使うと, 常に持ち上げたままの脚が基準高さを押し上げてしまう.
            if len(foot_ids) > 0:
                foot_z = robot.data.body_pos_w[:, foot_ids, 2]   # (N, F)
                if contact.shape[-1] == foot_z.shape[-1] and not torch.isnan(contact).any():
                    w = contact
                    nc = w.sum(dim=1, keepdim=True)
                    # 接地ゼロ(浮遊中)のときは最下点の足を地面とみなす
                    ref_z = torch.where(nc.squeeze(1) > 0,
                                        (foot_z * w).sum(dim=1) / nc.squeeze(1).clamp(min=1e-6),
                                        foot_z.min(dim=1).values)
                else:
                    ref_z = foot_z.mean(dim=1)
                base_h = robot.data.root_pos_w[:, 2] - ref_z
            else:
                base_h = robot.data.root_pos_w[:, 2]

            speed_xy = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

        timestep += 1
        time_sec = timestep * dt
        in_eval = timestep > args_cli.warmup_steps
        phase = "eval" if in_eval else "warmup"

        # warmup を過ぎた時点で物理レート積算を開始する
        if physics_metrics_ok and in_eval and not ACC["active"]:
            ACC["active"] = True
            print(f"[INFO] t={time_sec:.2f}s: 物理レート積算を開始しました.")

        # ---------------- 追従カメラ (ヨー角追従モードのみ手動更新) ----------------
        if args_cli.follow_cam and args_cli.cam_follow_heading:
            with torch.inference_mode():
                root_p = robot.data.root_pos_w[cam_env].cpu().numpy()
                _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[cam_env: cam_env + 1])
                yaw = float(torch.as_tensor(yaw)[0])
            az = math.radians(args_cli.cam_azimuth)
            d = args_cli.cam_distance
            lx, ly = -d * math.cos(az), -d * math.sin(az)
            # ロボット座標系のオフセットをヨー角で世界座標へ回転
            wx = math.cos(yaw) * lx - math.sin(yaw) * ly
            wy = math.sin(yaw) * lx + math.cos(yaw) * ly
            eye = np.array([root_p[0] + wx, root_p[1] + wy, root_p[2] + args_cli.cam_height])
            target = np.array([root_p[0], root_p[1], root_p[2] + args_cli.cam_lookat_z])
            # 指数移動平均で平滑化 (接地ごとの上下動でカメラが揺れるのを防ぐ)
            a = float(np.clip(args_cli.cam_smooth, 0.0, 0.999))
            if cam_state["eye"] is None:
                cam_state["eye"], cam_state["target"] = eye, target
            else:
                cam_state["eye"] = a * cam_state["eye"] + (1.0 - a) * eye
                cam_state["target"] = a * cam_state["target"] + (1.0 - a) * target
            _set_camera(env, cam_state["eye"], cam_state["target"])

        # ---------------- 時系列 CSV (1環境のみ) ----------------
        e = log_env
        row = [f"{time_sec:.6f}", phase]
        row += [f"{v:.6f}" for v in cmd[e].tolist()]
        row += [f"{v:.6f}" for v in lin_b[e].tolist()] + [f"{v:.6f}" for v in ang_b[e].tolist()]
        row += [f"{float(roll[e]):.6f}", f"{float(pitch[e]):.6f}", f"{float(base_h[e]):.6f}"]
        row += [f"{v:.6f}" for v in q[e].tolist()]
        row += [f"{v:.6f}" for v in qd[e].tolist()]
        row += [f"{v:.6f}" for v in tau[e].tolist()]
        row += [f"{v:.6f}" for v in p_joint[e].tolist()]
        row += [f"{float(p_pos[e]):.6f}", f"{float(p_abs[e]):.6f}"]
        row += [f"{v:.0f}" for v in contact[e].tolist()]
        writer.writerow(row)

        # ---------------- 集計バッファ (全環境) ----------------
        if in_eval:
            buf["tau"].append(tau.cpu().numpy())
            buf["qd"].append(qd.cpu().numpy())
            buf["p_pos"].append(p_pos.cpu().numpy())
            buf["p_abs"].append(p_abs.cpu().numpy())
            buf["speed_xy"].append(speed_xy.cpu().numpy())
            buf["vx_b"].append(lin_b[:, 0].cpu().numpy())
            buf["wz_b"].append(ang_b[:, 2].cpu().numpy())
            buf["cmd"].append(cmd.cpu().numpy())
            buf["roll"].append(roll.cpu().numpy())
            buf["pitch"].append(pitch.cpu().numpy())
            buf["base_h"].append(base_h.cpu().numpy())
            buf["contact"].append(contact.cpu().numpy())
            buf["drag"].append(drag.cpu().numpy())
            if grf_f is not None:
                buf["grf_f"].append(grf_f.cpu().numpy())
                buf["grf_r"].append(grf_r.cpu().numpy())

        # 終了条件
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if total_steps > 0 and timestep >= total_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    ACC["active"] = False
    try:
        env.unwrapped.sim.remove_physics_callback("eval_power_accum")
    except Exception:
        pass

    csv_file.close()
    print(f"[INFO] 時系列ログを保存しました: {ts_path}")

    # ========================================================================
    # 集計
    # ========================================================================
    if len(buf["tau"]) < 2:
        print("[WARN] 評価ステップが不足しているため集計をスキップします.")
    else:
        tau = np.stack(buf["tau"])            # (T, N, J)
        qd = np.stack(buf["qd"])              # (T, N, J)
        p_pos = np.stack(buf["p_pos"])        # (T, N)
        p_abs = np.stack(buf["p_abs"])        # (T, N)
        speed = np.stack(buf["speed_xy"])     # (T, N)
        vx_b = np.stack(buf["vx_b"])          # (T, N)
        wz_b = np.stack(buf["wz_b"])          # (T, N)
        cmd = np.stack(buf["cmd"])            # (T, N, 3)
        roll = _wrap_to_pi(np.stack(buf["roll"]))
        pitch = _wrap_to_pi(np.stack(buf["pitch"]))
        base_h = np.stack(buf["base_h"])      # (T, N)
        contact = np.stack(buf["contact"])    # (T, N, F)

        T = tau.shape[0]
        duration = T * dt

        # --- 正規化スケール ---
        if args_cli.leg_length is not None:
            leg_len = float(args_cli.leg_length)
            leg_src = "user-specified (--leg_length)"
        else:
            leg_len = float(np.nanmean(base_h))
            leg_src = "auto: 立位股関節高さの実測平均"
        print(f"\n[INFO] 正規化脚長 l = {leg_len:.4f} m  [{leg_src}]")
        if args_cli.leg_length is None:
            print("[WARN] l をモデルごとの実測値で正規化しています. モデル間比較では "
                  "--leg_length 0.4 のように共通の固定値を指定してください.")
        weight = masses * GRAVITY                     # (N,) [N] 環境ごとの実重量
        tau_scale = weight * leg_len                  # (N,) [N*m]

        distance = speed.sum(axis=0) * dt             # (N,) [m] 経路長
        valid = distance > 1e-3

        # --- 制御周期 (50 Hz) での推定値: 離散化誤差の比較対象として残す ---
        e_pos_ctrl = p_pos.sum(axis=0) * dt           # (N,) [J]
        e_abs_ctrl = p_abs.sum(axis=0) * dt
        cot_pos_ctrl = np.where(valid, e_pos_ctrl / (weight * np.maximum(distance, 1e-6)), np.nan)
        cot_abs_ctrl = np.where(valid, e_abs_ctrl / (weight * np.maximum(distance, 1e-6)), np.nan)
        tau_rms_ctrl = np.sqrt((tau ** 2).mean(axis=(0, 2)))

        # --- 物理レート (200 Hz) の積算値を優先して使う ---
        n_sub = int(ACC["n"])
        use_phys = physics_metrics_ok and n_sub > 0
        if use_phys:
            T_phys = n_sub * phys_dt
            e_pos = ACC["e_pos"].cpu().numpy()
            e_abs = ACC["e_abs"].cpu().numpy()
            tau_rms_j = np.sqrt(ACC["tau_sq"].cpu().numpy() / max(T_phys, 1e-9))       # (N, J)
            tau_peak_j = ACC["tau_peak"].cpu().numpy()                                  # (N, J)
            sat = ACC["sat_curve"].cpu().numpy() / n_sub                                # (N, J)
            sat_const = ACC["sat_const"].cpu().numpy() / n_sub
            tau_rms_all = np.sqrt(ACC["tau_sq"].sum(dim=1).cpu().numpy()
                                  / (max(T_phys, 1e-9) * num_joints))
            tau_peak_all = tau_peak_j.max(axis=1)
            e_front = ACC["e_front"].cpu().numpy()
            e_rear = ACC["e_rear"].cpu().numpy()
            rate_note = f"physics-rate ({1/phys_dt:.0f} Hz, {n_sub} substeps)"
            print(f"[INFO] トルク・仕事率の集計に物理レート積算を使用: {n_sub} substeps "
                  f"= {T_phys:.2f} s (制御周期換算 {T:d} steps = {duration:.2f} s)")
        else:
            e_pos, e_abs = e_pos_ctrl, e_abs_ctrl
            tau_rms_j = np.sqrt((tau ** 2).mean(axis=0))
            tau_peak_j = np.abs(tau).max(axis=0)
            lim = effort_limits.detach().cpu().numpy()
            if lim.shape != (tau.shape[1], tau.shape[2]):
                lim = np.broadcast_to(lim.reshape(-1)[-tau.shape[2]:], (tau.shape[1], tau.shape[2]))
            with np.errstate(invalid="ignore"):
                sat = (np.abs(tau) >= 0.99 * lim[None, :, :]).mean(axis=0)
            sat = np.where(np.isnan(lim), np.nan, sat)
            sat_const = sat
            tau_rms_all = tau_rms_ctrl
            tau_peak_all = np.abs(tau).max(axis=(0, 2))
            if len(front_j) > 0 and len(rear_j) > 0:
                e_front = np.clip(tau[:, :, front_j] * qd[:, :, front_j], 0, None).sum(axis=(0, 2)) * dt
                e_rear = np.clip(tau[:, :, rear_j] * qd[:, :, rear_j], 0, None).sum(axis=(0, 2)) * dt
            else:
                e_front = e_rear = np.full(tau.shape[1], np.nan)
            rate_note = f"control-rate ({1/dt:.0f} Hz)"
            print("[WARN] 物理レート積算が使えなかったため, 制御周期の値を使用します.")

        cot_pos = np.where(valid, e_pos / (weight * np.maximum(distance, 1e-6)), np.nan)
        cot_abs = np.where(valid, e_abs / (weight * np.maximum(distance, 1e-6)), np.nan)

        # 離散化誤差の大きさ (この値自体を limitation として報告できる)
        with np.errstate(invalid="ignore", divide="ignore"):
            cot_aliasing_err = np.where(np.abs(cot_pos) > 1e-9,
                                        (cot_pos_ctrl - cot_pos) / cot_pos, np.nan)

        # --- 足先以外の接地(引きずり)の時間割合 ---
        if len(buf["drag"]) == T:
            drag_frac = np.stack(buf["drag"]).mean(axis=0)
        else:
            drag_frac = np.full(tau.shape[1], np.nan)

        # --- 鉛直床反力の力積比 (生物力学で前後肢荷重分担に使う標準的な量) ---
        if len(buf["grf_f"]) == T and len(buf["grf_r"]) == T:
            imp_f = np.stack(buf["grf_f"]).sum(axis=0)     # (N,) [N*s]
            imp_r = np.stack(buf["grf_r"]).sum(axis=0)
            grf_front_frac = imp_f / np.maximum(imp_f + imp_r, 1e-9)
        else:
            grf_front_frac = np.full(tau.shape[1], np.nan)

        # --- 前後脚の分担 ---
        if len(front_j) > 0 and len(rear_j) > 0:
            fr_ratio = e_front / np.maximum(e_front + e_rear, 1e-9)
            tau_rms_front = np.sqrt((tau_rms_j[:, front_j] ** 2).mean(axis=1))
            tau_rms_rear = np.sqrt((tau_rms_j[:, rear_j] ** 2).mean(axis=1))
        else:
            fr_ratio = np.full(tau.shape[1], np.nan)
            tau_rms_front = tau_rms_rear = np.full(tau.shape[1], np.nan)

        # --- 歩容 ---
        if not np.isnan(contact).all():
            contact_b = _debounce_contacts(contact > 0.5, args_cli.contact_min_steps)
            duty = contact_b.mean(axis=0)                       # (N, F)
            onsets = _count_contact_onsets(contact_b)           # (N, F)
            stride_freq = onsets / duration                     # (N, F) [Hz]
            # 脚間の中央値を使う: 接地しない脚 (duty~0) が算術平均を壊すため.
            # ストライド周波数は本来 1 本の基準肢の周期で定義される量.
            mean_stride_freq = np.nanmedian(stride_freq, axis=1)     # (N,)
            stride_freq_mean_legs = np.nanmean(stride_freq, axis=1)  # 参考値
            stride_freq_spread = stride_freq.max(axis=1) - stride_freq.min(axis=1)
            mean_duty = np.nanmean(duty, axis=1)                # (N,)
            mean_speed = speed.mean(axis=0)
            stride_len = np.where(mean_stride_freq > 1e-6, mean_speed / np.maximum(mean_stride_freq, 1e-6), np.nan)
            stride_len_norm = stride_len / leg_len
            flight_frac = (contact_b.sum(axis=2) == 0).mean(axis=0)   # (N,) 全脚浮遊の時間割合
            duty_asym = duty.max(axis=1) - duty.min(axis=1)           # (N,) 脚間の duty 非対称性

            # --- 歩容解析 (機能している脚のみ, 環境ごと) ---
            gait, leg_phase = _gait_analysis(contact_b, foot_names, dt, mean_speed,
                                             leg_len, args_cli.live_duty_threshold)
            stride_len_norm = gait["stride_length_normalized"]
            mean_stride_freq = gait["stride_frequency_hz"]
        else:
            duty = np.full((tau.shape[1], 1), np.nan)
            mean_duty = np.full(tau.shape[1], np.nan)
            mean_stride_freq = np.full(tau.shape[1], np.nan)
            stride_freq_mean_legs = np.full(tau.shape[1], np.nan)
            stride_freq_spread = np.full(tau.shape[1], np.nan)
            flight_frac = np.full(tau.shape[1], np.nan)
            duty_asym = np.full(tau.shape[1], np.nan)
            gait = {k: np.full(tau.shape[1], np.nan) for k in
                    ["duty_live", "flight_live", "n_live_legs", "gait_phase_R", "gait_phase_p",
                     "hind_phase_offset", "fore_hind_phase_offset", "stride_period_s"]}
            leg_phase = np.full((tau.shape[1], 4), np.nan)
            stride_len_norm = np.full(tau.shape[1], np.nan)
            mean_speed = speed.mean(axis=0)

        # --- 速度追従 ---
        err_vx = vx_b - cmd[:, :, 0]
        err_wz = wz_b - cmd[:, :, 2]
        rmse_vx = np.sqrt(np.nanmean(err_vx ** 2, axis=0))
        rmse_wz = np.sqrt(np.nanmean(err_wz ** 2, axis=0))

        # --- 安定性 ---
        # 0 基準の RMS は「傾いたまま走っている」定常オフセットと「揺れている」振動を
        # 混同する. 平均 (姿勢オフセット) と標準偏差 (動揺) に分けて報告する.
        roll_rms = np.sqrt((roll ** 2).mean(axis=0))
        pitch_rms = np.sqrt((pitch ** 2).mean(axis=0))
        roll_mean = roll.mean(axis=0)
        pitch_mean = pitch.mean(axis=0)
        roll_sd = roll.std(axis=0)
        pitch_sd = pitch.std(axis=0)
        base_h_std = base_h.std(axis=0)
        base_h_std_norm = base_h_std / leg_len

        # --- Froude 数 ---
        froude = mean_speed ** 2 / (GRAVITY * max(leg_len, 1e-6))

        # ------------------------------------------------------------------
        metrics = {
            "mean_speed_mps": mean_speed,
            "froude_number": froude,
            "cot_positive_work": cot_pos,
            "cot_absolute_work": cot_abs,
            "cot_positive_ctrl_rate": cot_pos_ctrl,
            "cot_aliasing_error": cot_aliasing_err,
            "tau_rms_Nm": tau_rms_all,
            "tau_peak_Nm": tau_peak_all,
            "tau_rms_normalized": tau_rms_all / tau_scale,
            "tau_peak_normalized": tau_peak_all / tau_scale,
            "tau_rms_front_Nm": tau_rms_front,
            "tau_rms_rear_Nm": tau_rms_rear,
            "front_work_fraction": fr_ratio,
            "grf_front_impulse_fraction": grf_front_frac,
            "torque_saturation_ratio": sat.mean(axis=1),
            "torque_saturation_const_limit": sat_const.mean(axis=1),
            "duty_factor": mean_duty,
            "duty_factor_live_legs": gait["duty_live"],
            "n_live_legs": gait["n_live_legs"],
            "duty_asymmetry": duty_asym,
            "nonfoot_contact_fraction": drag_frac,
            "gait_phase_concentration_R": gait["gait_phase_R"],
            "gait_phase_rayleigh_p": gait["gait_phase_p"],
            "hind_pair_phase_offset": gait["hind_phase_offset"],
            "fore_hind_phase_offset": gait["fore_hind_phase_offset"],
            "stride_period_s": gait["stride_period_s"],
            "flight_phase_fraction": flight_frac,
            "flight_fraction_live_legs": gait["flight_live"],
            "stride_frequency_hz": mean_stride_freq,
            "stride_frequency_leg_spread": stride_freq_spread,
            "stride_length_normalized": stride_len_norm,
            "rmse_lin_vel_x": rmse_vx,
            "rmse_ang_vel_z": rmse_wz,
            "roll_rms_rad": roll_rms,
            "pitch_rms_rad": pitch_rms,
            "roll_offset_rad": roll_mean,
            "pitch_offset_rad": pitch_mean,
            "roll_sd_rad": roll_sd,
            "pitch_sd_rad": pitch_sd,
            "base_height_mean_m": base_h.mean(axis=0),
            "base_height_std_normalized": base_h_std_norm,
        }

        summary = {
            "tag": args_cli.eval_tag,
            "checkpoint": resume_path,
            "task": args_cli.task,
            "num_envs": int(num_envs),
            "eval_steps": int(T),
            "dt": float(dt),
            "duration_s": float(duration),
            "total_mass_kg": total_mass,
            "leg_length_m": float(leg_len),
            "leg_length_source": leg_src,
            "metric_rate": rate_note,
            "physics_dt": float(phys_dt),
            "decimation": int(decim),
            "physics_substeps_accumulated": int(ACC["n"]),
            "torque_scale_mgl_Nm": float(np.mean(tau_scale)),
            "mass_std_kg": float(masses.std()),
        }

        print("\n" + "=" * 74)
        print(f" 評価結果  tag={args_cli.eval_tag}   n_env={num_envs}   T={T} steps ({duration:.2f} s)")
        print(f" 質量 m={total_mass:.3f} kg,  正規化脚長 l={leg_len:.4f} m,  mgl={np.mean(tau_scale):.3f} N*m")
        print("=" * 74)
        print(f"{'指標':<32s}{'mean':>13s}{'std':>13s}{'単位':>12s}")
        print("-" * 74)
        units = {
            "mean_speed_mps": "m/s", "froude_number": "-", "cot_positive_work": "-",
            "cot_absolute_work": "-", "tau_rms_Nm": "N*m", "tau_peak_Nm": "N*m",
            "tau_rms_normalized": "-", "tau_peak_normalized": "-",
            "tau_rms_front_Nm": "N*m", "tau_rms_rear_Nm": "N*m",
            "front_work_fraction": "-", "grf_front_impulse_fraction": "-",
            "torque_saturation_ratio": "-",
            "duty_factor": "-", "duty_factor_live_legs": "-", "n_live_legs": "legs",
            "duty_asymmetry": "-", "flight_phase_fraction": "-", "flight_fraction_live_legs": "-",
            "nonfoot_contact_fraction": "-", "gait_phase_concentration_R": "-",
            "gait_phase_rayleigh_p": "-", "hind_pair_phase_offset": "cycle",
            "fore_hind_phase_offset": "cycle", "stride_period_s": "s",
            "stride_frequency_hz": "Hz", "stride_frequency_leg_spread": "Hz",
            "cot_positive_ctrl_rate": "-", "cot_aliasing_error": "-",
            "torque_saturation_const_limit": "-",
            "roll_offset_rad": "rad", "pitch_offset_rad": "rad",
            "roll_sd_rad": "rad", "pitch_sd_rad": "rad",
            "stride_length_normalized": "-", "rmse_lin_vel_x": "m/s",
            "rmse_ang_vel_z": "rad/s", "roll_rms_rad": "rad", "pitch_rms_rad": "rad",
            "base_height_mean_m": "m", "base_height_std_normalized": "-",
        }
        for k, v in metrics.items():
            m = float(np.nanmean(v))
            s = float(np.nanstd(v))
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"] = s
            print(f"{k:<32s}{m:>13.5f}{s:>13.5f}{units.get(k, ''):>12s}")
        print("=" * 74 + "\n")

        # 関節ごとのトルク内訳
        per_joint = {}
        for j, n in enumerate(joint_names):
            per_joint[n] = {
                "tau_rms_Nm": float(np.nanmean(tau_rms_j[:, j])),
                "tau_peak_Nm": float(np.nanmean(tau_peak_j[:, j])),
                "saturation_ratio": float(np.nanmean(sat[:, j])),
            }
        summary["per_joint"] = per_joint
        if not np.isnan(duty).all():
            summary["per_foot_duty_factor"] = {
                foot_names[i]: float(np.nanmean(duty[:, i])) for i in range(duty.shape[1])
            }

        json_path = os.path.join(log_dir, f"eval_{args_cli.eval_tag}_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(log_dir, f"eval_{args_cli.eval_tag}_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std", "unit"])
            for k, v in metrics.items():
                w.writerow([k, float(np.nanmean(v)), float(np.nanstd(v)), units.get(k, "")])
            w.writerow([])
            w.writerow(["joint", "tau_rms_Nm", "tau_peak_Nm", "saturation_ratio"])
            for n, d in per_joint.items():
                w.writerow([n, d["tau_rms_Nm"], d["tau_peak_Nm"], d["saturation_ratio"]])

        # 環境ごとの生値 (seed 間比較や統計検定に使う)
        per_env_path = os.path.join(log_dir, f"eval_{args_cli.eval_tag}_per_env.csv")
        with open(per_env_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["env_id"] + list(metrics.keys()))
            for i in range(num_envs):
                w.writerow([i] + [float(metrics[k][i]) for k in metrics])

        print(f"[INFO] 集計指標を保存しました:\n  {json_path}\n  {csv_path}\n  {per_env_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
