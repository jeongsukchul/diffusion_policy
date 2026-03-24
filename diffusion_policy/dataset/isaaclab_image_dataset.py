from pathlib import Path
from typing import Dict, List, Optional, Sequence
import copy

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class IsaacLabImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys: Optional[List[str]] = None,
        obs_group: str = "teleop_obs",
        action_key: str = "actions",
        rgb_root: Optional[str] = None,
        image_shape: Optional[Sequence[int]] = None,
        n_obs_steps=None,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()

        if obs_keys is None:
            obs_keys = [
                "ee_pose_in_robot_frame",
                "joint_pos",
                "joint_vel",
                "object_position",
                "target_object_position",
                "gripper_object_contact",
            ]

        self.dataset_path = str(dataset_path)
        self.obs_keys = list(obs_keys)
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.n_obs_steps = n_obs_steps
        self.image_root = _resolve_image_root(
            dataset_path=self.dataset_path,
            rgb_root=rgb_root,
        )
        image_episode_dirs = _get_image_episode_dirs(self.image_root)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        episode_image_paths = list()
        with h5py.File(self.dataset_path) as file:
            demos = file["data"]
            lexicographic_demo_names = sorted(demos.keys())
            demo_name_to_image_episode_idx = {
                demo_name: idx for idx, demo_name in enumerate(lexicographic_demo_names)
            }
            for episode_idx in tqdm(
                range(len(demos)),
                desc="Loading IsaacLab visuomotor hdf5",
            ):
                demo_name = f"demo_{episode_idx}"
                demo = demos[demo_name]
                episode = _demo_to_episode(
                    demo=demo,
                    obs_group=obs_group,
                    obs_keys=self.obs_keys,
                    action_key=action_key,
                )

                frame_paths = _load_episode_frame_paths(
                    episode_dir=image_episode_dirs[demo_name_to_image_episode_idx[demo_name]],
                    episode_idx=episode_idx,
                    episode_length=len(episode["action"]),
                )

                replay_buffer.add_episode(episode)
                episode_image_paths.append(frame_paths)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k["state"] = n_obs_steps

        self.replay_buffer = replay_buffer
        self.episode_image_paths = episode_image_paths
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=["state", "action"],
            key_first_k=key_first_k,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_set.train_mask,
            keys=["state", "action"],
            key_first_k={"state": self.n_obs_steps} if self.n_obs_steps is not None else dict(),
        )
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer["action"])
        )
        normalizer["state"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer["state"])
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        image = self._sample_images(idx)
        t_slice = slice(self.n_obs_steps)

        data = {
            "obs": {
                "image": image[t_slice].astype(np.float32) / 255.0,
                "state": sample["state"][t_slice].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)

    def _sample_images(self, idx: int) -> np.ndarray:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        episode_idx = np.searchsorted(self.replay_buffer.episode_ends, buffer_start_idx, side="right")
        episode_start = 0
        if episode_idx > 0:
            episode_start = int(self.replay_buffer.episode_ends[episode_idx - 1])

        frame_paths = self.episode_image_paths[episode_idx]
        local_start_idx = int(buffer_start_idx - episode_start)
        local_end_idx = int(buffer_end_idx - episode_start)
        sequence_paths = list(frame_paths[local_start_idx:local_end_idx])

        images = [_read_image(path, image_shape=self.image_shape) for path in sequence_paths]
        if sample_start_idx > 0:
            first_image = images[0]
            for _ in range(sample_start_idx):
                images.insert(0, first_image.copy())
        if sample_end_idx < self.horizon:
            last_image = images[-1]
            for _ in range(self.horizon - sample_end_idx):
                images.append(last_image.copy())

        return np.stack(images, axis=0)


def _demo_to_episode(demo, obs_group: str, obs_keys: List[str], action_key: str):
    raw_obs = demo[obs_group]
    state = np.concatenate([raw_obs[key][:] for key in obs_keys], axis=-1).astype(np.float32)
    action = demo[action_key][:].astype(np.float32)
    return {
        "state": state,
        "action": action,
    }


def _resolve_image_root(dataset_path: str, rgb_root: Optional[str]) -> Path:
    if rgb_root is not None:
        root = Path(rgb_root).expanduser()
        dataset_name = Path(dataset_path).name
        if root.name.startswith(dataset_name) and root.is_dir():
            return root

        candidates = sorted(path for path in root.glob(f"{dataset_name}_*") if path.is_dir())
        if len(candidates) > 0:
            return candidates[-1]
        raise FileNotFoundError(
            f"Could not find extracted IsaacLab image folder for {dataset_name} under {root}"
        )

    dataset_file = Path(dataset_path).expanduser()
    default_root = dataset_file.parent / "Images"
    candidates = sorted(
        path for path in default_root.glob(f"{dataset_file.name}_*") if path.is_dir()
    )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not auto-discover extracted IsaacLab image folder under {default_root}"
        )
    return candidates[-1]


def _get_image_episode_dirs(image_root: Path) -> List[Path]:
    episode_dirs = sorted(
        (path for path in image_root.glob("ep*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("ep")),
    )
    if len(episode_dirs) == 0:
        raise FileNotFoundError(f"No episode image directories found under {image_root}")
    return episode_dirs


def _load_episode_frame_paths(
    episode_dir: Path,
    episode_idx: int,
    episode_length: int,
) -> List[str]:
    frame_paths = sorted(str(path) for path in episode_dir.glob("frame_*.png"))
    if len(frame_paths) == episode_length + 1:
        frame_paths = frame_paths[:-1]
    elif len(frame_paths) != episode_length:
        raise ValueError(
            f"Episode {episode_idx} frame count mismatch: found {len(frame_paths)} pngs in "
            f"{episode_dir}, expected {episode_length}"
        )
    return frame_paths


def _read_image(image_path: str, image_shape: Optional[Sequence[int]]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image_shape is not None:
            _, height, width = image_shape
            image = image.resize((width, height), Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.uint8)
    return np.moveaxis(image_array, -1, 0)
