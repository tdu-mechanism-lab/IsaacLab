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
parser.add_argument("--contact_threshold", type=float, default=1.0, help="接地判定の力しきい値 [N].")
parser.add_argument("--foot_pattern", type=str, default=".*_foot", help="足先ボディ名の正規表現.")
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


def _get_effort_limits(robot, num_joints, device):
    """関節の effort limit を取得. API 名がバージョンで違うため順に試す."""
    for attr in ("joint_effort_limits", "joint_effort_limit", "joint_effort_limit_sim"):
        val = getattr(robot.data, attr, None)
        if val is not None:
            return val.clone().to(device)
    try:
        return robot.root_physx_view.get_dof_max_forces().to(device)
    except Exception:
        print("[WARN] effort limit を取得できませんでした. 飽和率は NaN になります.")
        return torch.full((1, num_joints), float("nan"), device=device)


def _wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


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
    if contact_sensor is not None and len(foot_ids) > 0:
        sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(args_cli.foot_pattern)
        print(f"[INFO] Contact sensor foot bodies: {sensor_foot_names}")
    if contact_sensor is None:
        print("[WARN] 接触センサが見つかりません. duty factor / ストライド周波数は NaN になります.")

    # --- 前脚 / 後脚の関節インデックス (URDF の F*/R* 命名を前提) ---
    front_j = [i for i, n in enumerate(joint_names) if n.startswith("F")]
    rear_j = [i for i, n in enumerate(joint_names) if n.startswith("R")]
    print(f"[INFO] Front joints ({len(front_j)}): {[joint_names[i] for i in front_j]}")
    print(f"[INFO] Rear  joints ({len(rear_j)}): {[joint_names[i] for i in rear_j]}")

    # --- 質量・脚長 (正規化用) ---
    total_mass = float(robot.data.default_mass[0].sum().item())
    effort_limits = _get_effort_limits(robot, num_joints, device)
    if effort_limits.dim() == 1:
        effort_limits = effort_limits.unsqueeze(0)

    print(f"[INFO] Total mass = {total_mass:.4f} kg")
    print(f"[INFO] Effort limit (first joint) = {float(effort_limits[0, 0].item()):.3f} N*m")

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
                           "cmd", "roll", "pitch", "base_h", "contact"]}

    # reset environment
    obs = env.get_observations()
    timestep = 0
    log_env = min(args_cli.log_env, num_envs - 1)
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

            # 胴体高さ: 接地足の平均高さからの相対値 (不整地でも意味を持つ)
            if len(foot_ids) > 0:
                foot_z = robot.data.body_pos_w[:, foot_ids, 2]   # (N, F)
                base_h = robot.data.root_pos_w[:, 2] - foot_z.mean(dim=1)
            else:
                base_h = robot.data.root_pos_w[:, 2]

            # 速度コマンド
            if command_name is not None:
                cmd = env.unwrapped.command_manager.get_command(command_name)[:, :3]
            else:
                cmd = torch.full((num_envs, 3), float("nan"), device=device)

            # 接地判定
            if contact_sensor is not None and len(sensor_foot_ids) > 0:
                net_f = contact_sensor.data.net_forces_w[:, sensor_foot_ids, :]   # (N, F, 3)
                contact = (torch.norm(net_f, dim=-1) > args_cli.contact_threshold).float()
            else:
                contact = torch.full((num_envs, max(len(foot_ids), 1)), float("nan"), device=device)

            speed_xy = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

        timestep += 1
        time_sec = timestep * dt
        in_eval = timestep > args_cli.warmup_steps
        phase = "eval" if in_eval else "warmup"

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

        # 終了条件
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if total_steps > 0 and timestep >= total_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

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
        leg_len = args_cli.leg_length if args_cli.leg_length is not None else float(np.nanmean(base_h))
        weight = total_mass * GRAVITY                 # [N]
        tau_scale = weight * leg_len                  # [N*m]

        # --- エネルギー / CoT ---
        e_pos = p_pos.sum(axis=0) * dt                # (N,) [J]
        e_abs = p_abs.sum(axis=0) * dt
        distance = speed.sum(axis=0) * dt             # (N,) [m]
        valid = distance > 1e-3
        cot_pos = np.where(valid, e_pos / (weight * np.maximum(distance, 1e-6)), np.nan)
        cot_abs = np.where(valid, e_abs / (weight * np.maximum(distance, 1e-6)), np.nan)

        # --- トルク ---
        tau_rms_j = np.sqrt((tau ** 2).mean(axis=0))          # (N, J)
        tau_peak_j = np.abs(tau).max(axis=0)                  # (N, J)
        lim = effort_limits.cpu().numpy().reshape(1, 1, -1)   # (1, 1, J)
        sat = (np.abs(tau) >= 0.99 * lim).mean(axis=0)        # (N, J) 飽和時間割合

        tau_rms_all = np.sqrt((tau ** 2).mean(axis=(0, 2)))   # (N,)
        tau_peak_all = np.abs(tau).max(axis=(0, 2))           # (N,)

        # --- 前後脚の負荷分担 ---
        if len(front_j) > 0 and len(rear_j) > 0:
            e_front = np.clip(tau[:, :, front_j] * qd[:, :, front_j], 0, None).sum(axis=(0, 2)) * dt
            e_rear = np.clip(tau[:, :, rear_j] * qd[:, :, rear_j], 0, None).sum(axis=(0, 2)) * dt
            fr_ratio = e_front / np.maximum(e_front + e_rear, 1e-9)
            tau_rms_front = np.sqrt((tau[:, :, front_j] ** 2).mean(axis=(0, 2)))
            tau_rms_rear = np.sqrt((tau[:, :, rear_j] ** 2).mean(axis=(0, 2)))
        else:
            fr_ratio = np.full(tau.shape[1], np.nan)
            tau_rms_front = tau_rms_rear = np.full(tau.shape[1], np.nan)

        # --- 歩容 ---
        if not np.isnan(contact).all():
            contact_b = contact > 0.5
            duty = contact_b.mean(axis=0)                       # (N, F)
            onsets = _count_contact_onsets(contact_b)           # (N, F)
            stride_freq = onsets / duration                     # (N, F) [Hz]
            mean_stride_freq = np.nanmean(stride_freq, axis=1)  # (N,)
            mean_duty = np.nanmean(duty, axis=1)                # (N,)
            mean_speed = speed.mean(axis=0)
            stride_len = np.where(mean_stride_freq > 1e-6, mean_speed / np.maximum(mean_stride_freq, 1e-6), np.nan)
            stride_len_norm = stride_len / leg_len
        else:
            duty = np.full((tau.shape[1], 1), np.nan)
            mean_duty = np.full(tau.shape[1], np.nan)
            mean_stride_freq = np.full(tau.shape[1], np.nan)
            stride_len_norm = np.full(tau.shape[1], np.nan)
            mean_speed = speed.mean(axis=0)

        # --- 速度追従 ---
        err_vx = vx_b - cmd[:, :, 0]
        err_wz = wz_b - cmd[:, :, 2]
        rmse_vx = np.sqrt(np.nanmean(err_vx ** 2, axis=0))
        rmse_wz = np.sqrt(np.nanmean(err_wz ** 2, axis=0))

        # --- 安定性 ---
        roll_rms = np.sqrt((roll ** 2).mean(axis=0))
        pitch_rms = np.sqrt((pitch ** 2).mean(axis=0))
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
            "tau_rms_Nm": tau_rms_all,
            "tau_peak_Nm": tau_peak_all,
            "tau_rms_normalized": tau_rms_all / tau_scale,
            "tau_peak_normalized": tau_peak_all / tau_scale,
            "tau_rms_front_Nm": tau_rms_front,
            "tau_rms_rear_Nm": tau_rms_rear,
            "front_work_fraction": fr_ratio,
            "torque_saturation_ratio": sat.mean(axis=1),
            "duty_factor": mean_duty,
            "stride_frequency_hz": mean_stride_freq,
            "stride_length_normalized": stride_len_norm,
            "rmse_lin_vel_x": rmse_vx,
            "rmse_ang_vel_z": rmse_wz,
            "roll_rms_rad": roll_rms,
            "pitch_rms_rad": pitch_rms,
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
            "torque_scale_mgl_Nm": float(tau_scale),
        }

        print("\n" + "=" * 74)
        print(f" 評価結果  tag={args_cli.eval_tag}   n_env={num_envs}   T={T} steps ({duration:.2f} s)")
        print(f" 質量 m={total_mass:.3f} kg,  正規化脚長 l={leg_len:.4f} m,  mgl={tau_scale:.3f} N*m")
        print("=" * 74)
        print(f"{'指標':<32s}{'mean':>13s}{'std':>13s}{'単位':>12s}")
        print("-" * 74)
        units = {
            "mean_speed_mps": "m/s", "froude_number": "-", "cot_positive_work": "-",
            "cot_absolute_work": "-", "tau_rms_Nm": "N*m", "tau_peak_Nm": "N*m",
            "tau_rms_normalized": "-", "tau_peak_normalized": "-",
            "tau_rms_front_Nm": "N*m", "tau_rms_rear_Nm": "N*m",
            "front_work_fraction": "-", "torque_saturation_ratio": "-",
            "duty_factor": "-", "stride_frequency_hz": "Hz",
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
