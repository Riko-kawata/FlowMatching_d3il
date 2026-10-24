"""Inserting (gate insertion: 3 個のブロックを 3 つのゲートへ押し込む) の評価。

上流 D3IL に inserting 用の Sim が無いため Sorting_Sim を元に作成。
- 観測は Inserting_Dataset と同じ 13 次元: 目標ロボット xy | 実ロボット xy | ブロック 3 個 (xy, tan yaw)。
  env.get_observation() が後ろ 11 次元を返すので、目標位置を前に連結する。
- 行動は目標 xy の差分。z と姿勢は固定 (fixed_z, quat [0,1,0,0])。
- success は 3 個とも入ったとき。mode は入った順 (6 通り)。
  mode の事前分布ファイルは無いので KL は出さず、成功 ep の mode エントロピーを記録する。
"""
import logging
import os

import multiprocessing as mp
import random

import numpy as np
import torch
import wandb

from envs.gym_inserting_env.gym_inserting.envs.gate_insertion import Gate_Insertion_Env
from simulation.base_sim import BaseSim
from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Inserting_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 60,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False,
            max_steps_per_episode: int = 2000,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)
        assert not if_vision, "inserting には画像データが無い"

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context
        self.max_steps_per_episode = max_steps_per_episode

        self.test_contexts = np.load(sim_framework_path("environments/dataset/data/inserting/test_contexts.pkl"),
                                     allow_pickle=True)
        assert n_contexts <= len(self.test_contexts)
        self.n_mode = 6

    def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, successes, n_inserted, pid, cpu_set):
        assign_process_to_cpu(os.getpid(), cpu_set)
        torch.set_num_threads(1)

        env = Gate_Insertion_Env(max_steps_per_episode=self.max_steps_per_episode, render=self.render)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for context in contexts:
            for i in range(n_trajectories):
                agent.reset()
                obs = env.reset(random=False, context=self.test_contexts[context])

                pred_action = env.robot_state()
                fixed_z = pred_action[2:]
                done = False
                info = {}
                while not done:
                    obs = np.concatenate((pred_action[:2], obs))
                    pred_action = agent.predict(obs)
                    pred_action = pred_action[0] + obs[:2]
                    pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)
                    obs, reward, done, info = env.step(pred_action)

                mode_encoding[context, i] = torch.tensor(info['mode'])
                successes[context, i] = torch.tensor(float(info['success']))
                n_inserted[context, i] = torch.tensor(float(info['one_box_success'] + info['two_box_success']
                                                            + info['three_box_success']))

    def test_agent(self, agent):
        log.info('Starting trained model evaluation')

        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        n_inserted = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)
        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))
        _off = int(os.environ.get("D3IL_CPU_OFFSET", "0")) % num_cpu
        cpu_set = cpu_set[_off:] + cpu_set[:_off]

        ctx = mp.get_context('spawn')
        if self.n_cores > 1:
            # context を均等に分ける (n_contexts が n_cores で割り切れなくても落とさない)
            chunks = np.array_split(contexts, self.n_cores)
            p_list = []
            for i, ch in enumerate(chunks):
                if len(ch) == 0:
                    continue
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs=dict(agent=agent, contexts=ch, n_trajectories=self.n_trajectories_per_context,
                                mode_encoding=mode_encoding, successes=successes, n_inserted=n_inserted,
                                pid=i, cpu_set=set(cpu_set[i:i + 1])),
                )
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]
        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context, mode_encoding, successes,
                            n_inserted, 0, set(cpu_set[:1]))

        success_rate = torch.mean(successes).item()
        m = mode_encoding[successes == 1].long()
        if len(m):
            p = torch.bincount(m, minlength=self.n_mode + 1)[1:].float()
            p = p / p.sum()
            entropy = -(p * torch.log(p + 1e-12)).sum().item() / np.log(self.n_mode)
        else:
            entropy = 0.0
        mean_inserted = n_inserted.mean().item()

        wandb.log({'Metrics/successes': success_rate, 'Metrics/entropy': entropy,
                   'Metrics/mean_inserted': mean_inserted})
        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')
        print(f'mean_inserted {mean_inserted}')
        for k in (1, 2, 3):
            print(f'{k}_box_rate {(n_inserted >= k).float().mean().item()}')

        # エピソードごとの結果 (context x trajectory)。analysis/eval_vision_ckpt.py が npz に保存する
        self.last_results = {k: v.numpy().copy() for k, v in dict(successes=successes, n_inserted=n_inserted, mode=mode_encoding).items()}
        return success_rate, mode_encoding
