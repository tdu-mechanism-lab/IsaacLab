# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A manager-based RL environment that keeps a diverged environment from corrupting the policy update."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class SanitizedManagerBasedRLEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` whose step rewards and observations are finite and bounded.

    The rewards of a step are computed before the environments that the termination terms flagged are reset, so the
    step on which the simulation of an environment blows up still returns its huge rewards. This class replaces
    non-finite values by zero and clamps the step reward to ``cfg.step_reward_clip`` and the observations to
    ``cfg.step_obs_clip``. Both default to values far outside the normal range, so ordinary steps are unchanged.
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        r_max = getattr(self.cfg, "step_reward_clip", 5.0)
        o_max = getattr(self.cfg, "step_obs_clip", 1.0e3)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-r_max, r_max)
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                obs[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-o_max, o_max)
        return obs, rew, terminated, truncated, extras
