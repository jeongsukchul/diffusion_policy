from typing import List
import copy

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class IsaacLabLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys: List[str] = None,
        obs_group: str = "teleop_obs",
        action_key: str = "actions",
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        if obs_keys is None:
            obs_keys = [
                "ee_pose_in_robot_frame",
                "joint_pos",
                "joint_vel",
                "object_position",
                "target_object_position",
                "gripper_object_contact",
            ]
        obs_keys = list(obs_keys)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file["data"]
            for i in tqdm(range(len(demos)), desc="Loading IsaacLab hdf5 to ReplayBuffer"):
                demo = demos[f"demo_{i}"]
                episode = _demo_to_obs(
                    demo=demo,
                    obs_group=obs_group,
                    obs_keys=obs_keys,
                    action_key=action_key,
                )
                replay_buffer.add_episode(episode)

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

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()
        obs_stat = array_to_stats(self.replay_buffer["obs"])
        action_stat = array_to_stats(self.replay_buffer["action"])

        normalizer["obs"] = normalizer_from_stat(obs_stat)
        normalizer["action"] = get_identity_normalizer_from_stat(action_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx):
        data = self.sampler.sample_sequence(idx)
        return dict_apply(data, torch.from_numpy)


def _demo_to_obs(demo, obs_group: str, obs_keys: List[str], action_key: str):
    raw_obs = demo[obs_group]
    obs = np.concatenate([raw_obs[key][:] for key in obs_keys], axis=-1).astype(np.float32)
    action = demo[action_key][:].astype(np.float32)
    return {
        "obs": obs,
        "action": action,
    }


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )
