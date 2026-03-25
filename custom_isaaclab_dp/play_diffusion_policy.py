#!/usr/bin/env python3

"""Run a trained diffusion policy checkpoint in Isaac-Lift-Physics-Select-Cube-Franka-IK-Rel-v0."""

import argparse
import collections
import os
import pathlib
import random
import sys
import traceback
from typing import Dict, Iterable, List


def _default_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


REPO_ROOT = _default_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher


def _require_module(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing Python module '{module_name}' in the current IsaacLab environment.\n"
            f"Install it with: {install_hint}"
        ) from exc


parser = argparse.ArgumentParser(description="Play/evaluate a diffusion policy checkpoint in IsaacLab.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a diffusion policy .ckpt file. If omitted, the most recent latest.ckpt under data/outputs is used.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Physics-Select-Cube-Franka-IK-Rel-v0",
    help="IsaacLab task name.",
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--horizon", type=int, default=400, help="Maximum steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
parser.add_argument(
    "--policy_device",
    type=str,
    default=None,
    help="Torch device for policy inference. Defaults to IsaacLab's --device if omitted.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of IsaacLab envs. This script uses 1.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--video", action="store_true", default=False, help="Record an evaluation video.")
parser.add_argument("--video_length", type=int, default=400, help="Length of recorded video in steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default="data/outputs/isaaclab_videos",
    help="Directory for saved evaluation videos.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(2.3604, 0.0140, 0.9440),
    metavar=("X", "Y", "Z"),
    help="Viewport camera eye position.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=(1.4369, 0.0253, 0.5606),
    metavar=("X", "Y", "Z"),
    help="Viewport camera target position.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_require_module("dill", "pip install dill")
_require_module("omegaconf", "pip install omegaconf")
_require_module("hydra", "pip install hydra-core")

import dill
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _find_latest_checkpoint(repo_root: pathlib.Path) -> str:
    outputs_dir = repo_root / "data" / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    latest_candidates = list(outputs_dir.glob("**/checkpoints/latest.ckpt"))
    if latest_candidates:
        latest_path = max(latest_candidates, key=lambda p: p.stat().st_mtime)
        return str(latest_path.resolve())

    ckpt_candidates = list(outputs_dir.glob("**/checkpoints/*.ckpt"))
    if ckpt_candidates:
        latest_path = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
        return str(latest_path.resolve())

    raise FileNotFoundError(f"No checkpoints found under: {outputs_dir}")


def _resolve_checkpoint_path(cli_checkpoint: str) -> str:
    if cli_checkpoint is None:
        checkpoint_path = _find_latest_checkpoint(REPO_ROOT)
    else:
        checkpoint_path = os.path.abspath(os.path.expanduser(cli_checkpoint))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _peek_checkpoint_cfg(checkpoint_path: str):
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    return payload["cfg"]


def _shape_meta_obs_items(cfg) -> Dict:
    shape_meta = getattr(cfg, "shape_meta", None)
    if shape_meta is None:
        shape_meta = cfg.task.shape_meta
    return shape_meta["obs"]


def _is_image_policy(cfg) -> bool:
    obs_meta = _shape_meta_obs_items(cfg)
    return any(attr.get("type", "low_dim") == "rgb" for attr in obs_meta.values())


checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
checkpoint_cfg = _peek_checkpoint_cfg(checkpoint_path)
use_image_policy_checkpoint = _is_image_policy(checkpoint_cfg)
if args_cli.video or use_image_policy_checkpoint:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[play_diffusion_policy] IsaacLab app launched", flush=True)

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab.envs.mdp.observations import generated_commands, joint_pos_rel, joint_vel_rel
from isaaclab_tasks.manager_based.manipulation.lift_physics import mdp as lift_physics_mdp

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_workspace(checkpoint_path: str, device: torch.device):
    print(f"[play_diffusion_policy] Loading checkpoint payload: {checkpoint_path}", flush=True)
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    print(f"[play_diffusion_policy] Instantiating policy: {cfg.policy._target_}", flush=True)
    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload["state_dicts"]
    if "ema_model" in state_dicts:
        policy.load_state_dict(state_dicts["ema_model"], strict=True)
    else:
        policy.load_state_dict(state_dicts["model"], strict=True)
    policy.to(device)
    policy.eval()
    print("[play_diffusion_policy] Policy loaded successfully", flush=True)
    return cfg, policy


def _extract_policy_obs(obs_dict):
    if "policy" in obs_dict and isinstance(obs_dict["policy"], dict):
        return obs_dict["policy"]
    return obs_dict


def _to_1d_tensor(x, device):
    if isinstance(x, torch.Tensor):
        out = x
    else:
        out = torch.as_tensor(x, dtype=torch.float32)
    out = out.to(device=device, dtype=torch.float32)
    if out.ndim > 1:
        out = out[0]
    return out.reshape(-1)


def _build_obs_vector(policy_obs: Dict, obs_keys: Iterable[str], device: torch.device) -> torch.Tensor:
    parts = []
    for key in obs_keys:
        if key not in policy_obs:
            raise KeyError(f"Missing observation key '{key}' in IsaacLab policy observations.")
        parts.append(_to_1d_tensor(policy_obs[key], device))
    return torch.cat(parts, dim=0)


def _build_isaaclab_lowdim_obs(base_env, obs_keys: Iterable[str], device: torch.device) -> torch.Tensor:
    value_map = {
        "ee_pose_in_robot_frame": lift_physics_mdp.ee_pose_in_robot_frame(
            base_env, frame_transformer_name="ee_frame", frame_name="end_effector"
        ),
        "joint_pos": joint_pos_rel(base_env),
        "joint_vel": joint_vel_rel(base_env),
        "object_position": lift_physics_mdp.object_position_in_robot_root_frame(base_env),
        "target_object_position": generated_commands(base_env, command_name="object_pose"),
        "gripper_object_contact": lift_physics_mdp.gripper_object_contact(base_env),
    }
    parts = []
    for key in obs_keys:
        if key not in value_map:
            raise KeyError(f"Unsupported IsaacLab observation key '{key}'.")
        parts.append(_to_1d_tensor(value_map[key], device))
    return torch.cat(parts, dim=0)


def _reset_history(obs_vec: torch.Tensor, n_obs_steps: int):
    history = collections.deque(maxlen=n_obs_steps)
    for _ in range(n_obs_steps):
        history.append(obs_vec.clone())
    return history


def _policy_input_from_history(history: collections.deque) -> torch.Tensor:
    return torch.stack(list(history), dim=0).unsqueeze(0)


def _reset_dict_history(obs_dict: Dict[str, torch.Tensor], n_obs_steps: int):
    history = {key: collections.deque(maxlen=n_obs_steps) for key in obs_dict.keys()}
    for _ in range(n_obs_steps):
        for key, value in obs_dict.items():
            history[key].append(value.clone())
    return history


def _policy_input_from_dict_history(history: Dict[str, collections.deque]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack(list(values), dim=0).unsqueeze(0) for key, values in history.items()}


def _append_dict_history(history: Dict[str, collections.deque], obs_dict: Dict[str, torch.Tensor]) -> None:
    for key, value in obs_dict.items():
        history[key].append(value.clone())


def _prepare_image_tensor(image, expected_shape, device: torch.device) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().to(device=device)
    else:
        tensor = torch.as_tensor(image, device=device)

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeeze, got shape {tuple(tensor.shape)}")

    expected_channels, expected_height, expected_width = tuple(expected_shape)
    if tensor.shape[-1] == expected_channels and tensor.shape[0] != expected_channels:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.shape[0] != expected_channels:
        raise ValueError(
            f"Could not match image channels for expected shape {tuple(expected_shape)} from input {tuple(tensor.shape)}"
        )

    tensor = tensor.to(dtype=torch.float32)
    if tensor.max().item() > 1.0 or tensor.min().item() < 0.0:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0.0, 1.0)

    if tensor.shape[1:] != (expected_height, expected_width):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(expected_height, expected_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return tensor.contiguous()


def _extract_rgb_tensor(policy_obs: Dict, env, rgb_key: str, expected_shape, device: torch.device) -> torch.Tensor:
    image = None
    if rgb_key in policy_obs:
        image = policy_obs[rgb_key]
    elif "image" in policy_obs:
        image = policy_obs["image"]
    else:
        rendered = env.render()
        if rendered is None:
            raise KeyError(
                f"Missing rgb observation '{rgb_key}' in env observations and env.render() returned None."
            )
        image = rendered
    return _prepare_image_tensor(image, expected_shape=expected_shape, device=device)


def _build_policy_obs_dict(cfg, env, obs_dict, obs_keys: Iterable[str], device: torch.device) -> Dict[str, torch.Tensor]:
    base_env = env.unwrapped
    policy_obs = _extract_policy_obs(obs_dict)
    obs_meta = _shape_meta_obs_items(cfg)
    result = dict()

    for key, attr in obs_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            result[key] = _extract_rgb_tensor(
                policy_obs=policy_obs,
                env=env,
                rgb_key=key,
                expected_shape=attr["shape"],
                device=device,
            )
            continue

        if key in policy_obs:
            result[key] = _to_1d_tensor(policy_obs[key], device)
            continue

        if key == "state":
            result[key] = _build_isaaclab_lowdim_obs(base_env, obs_keys, device)
            continue

        raise KeyError(f"Missing low-dimensional observation key '{key}' for image policy input.")

    return result


def _clip_action(action: torch.Tensor, action_space) -> torch.Tensor:
    low = torch.as_tensor(action_space.low, device=action.device, dtype=action.dtype).view(1, -1)
    high = torch.as_tensor(action_space.high, device=action.device, dtype=action.dtype).view(1, -1)
    return torch.max(torch.min(action, high), low)


def rollout(policy, env, obs_keys: List[str], n_obs_steps: int, horizon: int, success_term, device: torch.device):
    base_env = env.unwrapped
    obs_dict, _ = env.reset()
    cfg = getattr(policy, "_loaded_cfg")
    use_image_policy = _is_image_policy(cfg)

    if use_image_policy:
        current_obs = _build_policy_obs_dict(cfg, env, obs_dict, obs_keys, device)
        history = _reset_dict_history(current_obs, n_obs_steps)
    else:
        policy_obs = _extract_policy_obs(obs_dict)
        try:
            obs_vec = _build_obs_vector(policy_obs, obs_keys, device)
        except KeyError:
            obs_vec = _build_isaaclab_lowdim_obs(base_env, obs_keys, device)
        history = _reset_history(obs_vec, n_obs_steps)

    episode_reward = 0.0
    steps = 0
    success = False

    while steps < horizon and simulation_app.is_running():
        if use_image_policy:
            obs_batch = _policy_input_from_dict_history(history)
        else:
            obs_batch = _policy_input_from_history(history)
        with torch.no_grad():
            if use_image_policy:
                action_dict = policy.predict_action(obs_batch)
            else:
                action_dict = policy.predict_action({"obs": obs_batch})
        action_seq = action_dict["action"]

        for action_idx in range(action_seq.shape[1]):
            action = action_seq[:, action_idx, :]
            action = _clip_action(action, env.action_space)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(torch.as_tensor(reward).reshape(-1)[0].item())
            steps += 1
            if not getattr(args_cli, "headless", False):
                base_env.sim.render()

            if use_image_policy:
                current_obs = _build_policy_obs_dict(cfg, env, obs_dict, obs_keys, device)
                _append_dict_history(history, current_obs)
            else:
                policy_obs = _extract_policy_obs(obs_dict)
                try:
                    obs_vec = _build_obs_vector(policy_obs, obs_keys, device)
                except KeyError:
                    obs_vec = _build_isaaclab_lowdim_obs(base_env, obs_keys, device)
                history.append(obs_vec)

            if success_term is not None and bool(success_term.func(base_env, **success_term.params)[0]):
                success = True
                return success, episode_reward, steps
            if bool(torch.as_tensor(terminated).reshape(-1)[0].item()) or bool(
                torch.as_tensor(truncated).reshape(-1)[0].item()
            ):
                return success, episode_reward, steps
            if steps >= horizon:
                return success, episode_reward, steps

    return success, episode_reward, steps


def main():
    if args_cli.num_envs != 1:
        raise ValueError("This standalone runner currently supports only --num_envs 1.")
    print(f"[play_diffusion_policy] Using checkpoint: {checkpoint_path}", flush=True)

    policy_device_str = args_cli.policy_device if args_cli.policy_device is not None else args_cli.device
    device = torch.device(policy_device_str)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    cfg, policy = _load_workspace(checkpoint_path, device=device)
    policy._loaded_cfg = cfg
    obs_keys = list(cfg.task.dataset.obs_keys)
    n_obs_steps = int(cfg.n_obs_steps)
    print(f"[play_diffusion_policy] Obs keys: {obs_keys}", flush=True)
    print(f"[play_diffusion_policy] n_obs_steps: {n_obs_steps}", flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    print("[play_diffusion_policy] IsaacLab env cfg parsed", flush=True)
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None
    render_mode = "rgb_array" if (args_cli.video or use_image_policy_checkpoint) else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    print("[play_diffusion_policy] IsaacLab env created", flush=True)
    if args_cli.video:
        video_dir = os.path.abspath(os.path.expanduser(args_cli.video_dir))
        pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)
        env.metadata["render_fps"] = 10
        env.unwrapped.metadata["render_fps"] = 10
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[play_diffusion_policy] Video recording enabled: {video_dir}", flush=True)
    base_env = env.unwrapped

    base_env.seed(args_cli.seed)
    base_env.sim.set_camera_view(eye=tuple(args_cli.camera_eye), target=tuple(args_cli.camera_lookat))
    print("[play_diffusion_policy] Starting rollout loop", flush=True)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Task: {args_cli.task}")
    print(f"Obs keys: {obs_keys}")
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"Policy device: {device}")
    print(f"Success term available: {success_term is not None}")

    successes = 0
    rewards = []
    lengths = []

    for episode_idx in range(args_cli.num_episodes):
        episode_seed = args_cli.seed + episode_idx
        torch.manual_seed(episode_seed)
        np.random.seed(episode_seed)
        random.seed(episode_seed)
        base_env.seed(episode_seed)

        success, reward, length = rollout(
            policy=policy,
            env=env,
            obs_keys=obs_keys,
            n_obs_steps=n_obs_steps,
            horizon=args_cli.horizon,
            success_term=success_term,
            device=device,
        )
        successes += int(success)
        rewards.append(reward)
        lengths.append(length)
        print(
            f"[Episode {episode_idx + 1}/{args_cli.num_episodes}] "
            f"success={success} reward={reward:.4f} steps={length}"
        )

    env.close()

    success_rate = successes / max(args_cli.num_episodes, 1)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    print("")
    print(f"Success rate: {success_rate:.3f} ({successes}/{args_cli.num_episodes})")
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Mean episode length: {mean_length:.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[play_diffusion_policy] Unhandled exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
