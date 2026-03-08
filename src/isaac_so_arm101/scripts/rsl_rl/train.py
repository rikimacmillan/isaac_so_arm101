# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import statistics
import torch
import numpy as np
from collections import deque
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import store_code_state

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ==============================================================================
# DEBUG: DebugOnPolicyRunner — remove this class when training is stable.
# To revert: delete this class, then in main() change:
#   DebugOnPolicyRunner(...)  ->  OnPolicyRunner(...)
# ==============================================================================
class DebugOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with per-iteration diagnostics.

    Printed to stdout every iteration and written to TensorBoard under Debug/.

    What each section tells you
    ---------------------------
    Action norm        : should stay ~0.1-2.0. Above 5 = policy saturating outputs.
    Action delta norm  : should stay ~0.1-0.5. Above 3 = jerky step-to-step commands;
                         this is the direct cause of action_rate reward exploding.
    Per-joint range    : watch wrist_roll especially. Range > 3.5 rad = runaway joint.
    Obs abs max        : should stay <10 with empirical_normalization. Above 50 =
                         normaliser is failing and the network sees huge inputs.
    Value abs max      : early warning for crashes. 100-500 = yellow flag.
                         Above 500 = inf/nan crash within ~50 iterations.
    Gradient norms     : should stay <=1.0 with max_grad_norm=1.0. Consistently
                         above 1.0 = gradient clipping is not taking effect.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        # Per-step rollout buffers — populated inside learn(), read inside log()
        self._dbg_action_norms: list[float] = []
        self._dbg_action_delta_norms: list[float] = []
        self._dbg_joint_pos_mins: list[torch.Tensor] = []
        self._dbg_joint_pos_maxs: list[torch.Tensor] = []
        self._dbg_prev_actions: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Override learn() to inject per-step tracking into the rollout loop.
    # Everything outside the inner loop is identical to the RSL-RL source.
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            # ---- DEBUG: clear per-iteration rollout buffers ----
            self._dbg_action_norms.clear()
            self._dbg_action_delta_norms.clear()
            self._dbg_joint_pos_mins.clear()
            self._dbg_joint_pos_maxs.clear()
            self._dbg_prev_actions = None
            # ----------------------------------------------------

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # ---- DEBUG: track per-step action & joint stats ----
                    self._dbg_action_norms.append(actions.norm(dim=-1).mean().item())
                    if self._dbg_prev_actions is not None:
                        delta = (actions - self._dbg_prev_actions).norm(dim=-1)
                        self._dbg_action_delta_norms.append(delta.mean().item())
                    self._dbg_prev_actions = actions.clone()
                    # joint_pos is the first ObsTerm — first 6 dims.
                    # Order: base_yaw, shoulder_pitch, elbow_pitch,
                    #        wrist_pitch, wrist_roll, gripper_moving
                    # Adjust slice if your obs order differs.
                    jp = obs[:, :6]
                    self._dbg_joint_pos_mins.append(jp.min(dim=0).values.detach())
                    self._dbg_joint_pos_maxs.append(jp.max(dim=0).values.detach())
                    # ----------------------------------------------------

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ------------------------------------------------------------------
    # Override log() to append debug diagnostics after the standard output.
    # Uses the correct locs keys from the RSL-RL 3.x learn() locals().
    # ------------------------------------------------------------------
    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Run the parent log first so all standard RSL-RL output is preserved
        super().log(locs, width=width, pad=pad)

        it      = locs.get("it", 0)
        # last-step obs and actions are in locs (from the rollout loop locals)
        obs     = locs.get("obs")      # (num_envs, obs_dim) — last rollout step
        actions = locs.get("actions")  # (num_envs, act_dim) — last rollout step

        sep = "=" * width
        print(f"\n{sep}")
        print(f"  DEBUG  |  Iteration {it}")
        print(sep)

        # 1. Action norm across all rollout steps
        if self._dbg_action_norms:
            a = np.array(self._dbg_action_norms)
            w = "  <-- HIGH, policy saturating" if a.max() > 5.0 else ""
            print(f"  {'Action norm (mean/env per step)':.<{pad}} "
                  f"mean={a.mean():.4f}  max={a.max():.4f}  min={a.min():.4f}{w}")
        else:
            print(f"  {'Action norm':.<{pad}} (no data)")

        # 2. Action delta norm (step-to-step change — drives action_rate explosion)
        if self._dbg_action_delta_norms:
            a = np.array(self._dbg_action_delta_norms)
            w = "  <-- HIGH, action_rate will explode" if a.max() > 3.0 else ""
            print(f"  {'Action delta norm (step-to-step)':.<{pad}} "
                  f"mean={a.mean():.4f}  max={a.max():.4f}  min={a.min():.4f}{w}")

        # 3. Per-joint position range across the entire rollout
        if self._dbg_joint_pos_mins:
            jmin = torch.stack(self._dbg_joint_pos_mins).min(dim=0).values
            jmax = torch.stack(self._dbg_joint_pos_maxs).max(dim=0).values
            names = [
                "base_yaw", "shoulder_pitch", "elbow_pitch",
                "wrist_pitch", "wrist_roll", "gripper_moving",
            ]
            print(f"\n  Per-joint position range (obs space, normalised):")
            for i, name in enumerate(names):
                lo, hi = jmin[i].item(), jmax[i].item()
                rng = hi - lo
                w = "  *** RUNAWAY ***" if rng > 3.5 else ("  * wide *" if rng > 2.0 else "")
                print(f"    {name:.<22} [{lo:+.3f}, {hi:+.3f}]  range={rng:.3f}{w}")

        # 4. Last-step observation stats
        if obs is not None:
            omax  = obs.abs().max().item()
            nan_n = int(torch.isnan(obs).sum())
            inf_n = int(torch.isinf(obs).sum())
            w = "  <-- HIGH, normaliser may be failing" if omax > 50 else ""
            print(f"\n  Last-step observation stats:")
            print(f"    {'mean':.<{pad}} {obs.mean().item():+.4f}")
            print(f"    {'std':.<{pad}} {obs.std().item():.4f}")
            print(f"    {'abs max':.<{pad}} {omax:.4f}{w}")
            if nan_n: print(f"    *** {nan_n} NaN values in observations! ***")
            if inf_n: print(f"    *** {inf_n} Inf values in observations! ***")

        # 5. Last-step action stats
        if actions is not None:
            anorm = actions.norm(dim=-1)
            w = "  <-- HIGH" if anorm.max().item() > 5 else ""
            print(f"\n  Last-step action stats:")
            print(f"    {'norm mean':.<{pad}} {anorm.mean().item():.4f}")
            print(f"    {'norm max':.<{pad}} {anorm.max().item():.4f}{w}")
            print(f"    {'std':.<{pad}} {actions.std().item():.4f}")

        # 6. Loss dict — value_function_loss is the key crash indicator
        loss_dict = locs.get("loss_dict", {})
        if loss_dict:
            print(f"\n  Loss values:")
            for k, v in loss_dict.items():
                vf = float(v)
                w = ""
                if "value" in k and vf > 100:  w = "  <-- WARNING, diverging"
                if "value" in k and vf > 1000: w = "  <-- CRITICAL, crash imminent"
                if not np.isfinite(vf):         w = "  *** INF/NAN — crash this iter ***"
                print(f"    {k:.<{pad}} {vf:.6f}{w}")

        # 7. Gradient norms — self.alg.policy is the ActorCritic module in RSL-RL 3.x
        policy = getattr(self.alg, "policy", None)
        if policy is not None:
            # Try to get actor and critic sub-modules separately for finer resolution
            actor_mod  = getattr(policy, "actor",  None)
            critic_mod = getattr(policy, "critic", None)
            print(f"\n  Gradient norms (after PPO update):")
            if isinstance(actor_mod, torch.nn.Module):
                ag = self._grad_norm(actor_mod)
                aw = "  <-- HIGH, clipping may not be working" if ag > 1.0 else ""
                print(f"    {'actor':.<{pad}} {ag:.4f}{aw}")
            if isinstance(critic_mod, torch.nn.Module):
                cg = self._grad_norm(critic_mod)
                cw = "  <-- HIGH, value explosion risk" if cg > 1.0 else ""
                print(f"    {'critic':.<{pad}} {cg:.4f}{cw}")
            if not isinstance(actor_mod, torch.nn.Module) and not isinstance(critic_mod, torch.nn.Module):
                # Fall back to whole policy norm
                pg = self._grad_norm(policy)
                pw = "  <-- HIGH" if pg > 1.0 else ""
                print(f"    {'policy (combined)':.<{pad}} {pg:.4f}{pw}")
        else:
            print(f"\n  Gradient norms: (self.alg.policy not found)")

        print(sep + "\n")

        # 8. Mirror to TensorBoard under Debug/
        writer = getattr(self, "writer", None)
        if writer is not None:
            if self._dbg_action_norms:
                writer.add_scalar("Debug/action_norm_mean",  float(np.mean(self._dbg_action_norms)),  it)
                writer.add_scalar("Debug/action_norm_max",   float(np.max(self._dbg_action_norms)),   it)
            if self._dbg_action_delta_norms:
                writer.add_scalar("Debug/action_delta_mean", float(np.mean(self._dbg_action_delta_norms)), it)
                writer.add_scalar("Debug/action_delta_max",  float(np.max(self._dbg_action_delta_norms)),  it)
            if obs is not None:
                writer.add_scalar("Debug/obs_abs_max", obs.abs().max().item(), it)
                writer.add_scalar("Debug/obs_std",     obs.std().item(),       it)
            policy = getattr(self.alg, "policy", None)
            if policy is not None:
                actor_mod  = getattr(policy, "actor",  None)
                critic_mod = getattr(policy, "critic", None)
                if isinstance(actor_mod, torch.nn.Module):
                    writer.add_scalar("Debug/actor_grad_norm",  self._grad_norm(actor_mod),  it)
                if isinstance(critic_mod, torch.nn.Module):
                    writer.add_scalar("Debug/critic_grad_norm", self._grad_norm(critic_mod), it)

    @staticmethod
    def _grad_norm(module: torch.nn.Module) -> float:
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

# ==============================================================================
# END DEBUG
# ==============================================================================


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    # DEBUG: DebugOnPolicyRunner swapped in for OnPolicyRunner.
    # To revert, change DebugOnPolicyRunner back to OnPolicyRunner on the line below.
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = DebugOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()