import logging
import os
import pickle

import numpy as np
import torch

from environments.dataset.base_dataset import TrajectoryDataset
from environments.d3il.d3il_sim.utils.geometric_transformation import quat2euler
from agents.utils.sim_path import sim_framework_path


class Inserting_Dataset(TrajectoryDataset):
    """Gate insertion demonstrations (3 blocks into 3 gates).

    Observation layout (13 dims), matching what Inserting_Sim feeds the policy:
      desired robot xy | measured robot xy | per block: xy and tan(yaw)
    The env's own get_observation() supplies the last 11 (it starts from
    robot_state(), i.e. the MEASURED position); the simulation prepends the
    desired position, the same convention aligning and avoiding use.

    The action is the step-to-step delta of the desired cartesian position in xy
    only -- the demonstrations hold z (0.1224) and the orientation fixed.
    """

    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 13,
            action_dim: int = 2,
            max_len_data: int = 2400,
            window_size: int = 1,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        logging.info("Loading Inserting Dataset")

        inputs = []
        actions = []
        masks = []

        data_dir = sim_framework_path("environments/dataset/data/inserting/data_3")
        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:
            with open(os.path.join(data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]

            box_states = []
            for i in (1, 2, 3):
                box = env_state['box-%d' % i]
                box_pos = box['pos'][:, :2]
                # env.get_observation() uses tan of the yaw euler angle
                box_yaw = np.tan(quat2euler(box['quat'])[:, -1:])
                box_states += [box_pos, box_yaw]

            input_state = np.concatenate([robot_des_pos, robot_c_pos] + box_states, axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
            valid_len = min(len(vel_state), self.max_len_data)

            zero_obs[0, :valid_len, :] = input_state[:valid_len]
            zero_action[0, :valid_len, :] = vel_state[:valid_len]
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return obs, act, mask
