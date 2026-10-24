"""Energy Matching agent for D3IL (state input, action chunking).

robosuite_square の EM 実装 (flow.py / sample.py / train.py) を D3IL の
BaseAgent インターフェースに載せたもの。学習は Flow (v = -grad_x V(x|c) を flow matching で回帰) のみ。
評価は D3IL 標準の *_Sim.test_agent をそのまま使うので、他手法 (BESO / DDPM / ACT ...) と同じ条件で比較できる。

Square で得た教訓をそのまま引き継いでいる:
- ckpt 選択に valid flow を使わない (成功率と逆相関した)。評価は最終 ckpt。
- EMA の実効窓 1/(1-decay) は学習 step 数より十分短くする。
"""
import os
import math
import logging
from collections import deque

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from agents.base_agent import BaseAgent
from agents.models.em.em_policy import build_state_features

log = logging.getLogger(__name__)


class _EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)


class EMAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size: int,
            val_batch_size: int,
            num_workers: int,
            device: str,
            epoch: int,
            scale_data: bool,
            eval_every_n_epochs: int,
            obs_history: int = 2,
            horizon: int = 16,
            exec_horizon: int = 8,
            yaw_cols=(),
            drop_cols=(),
            aligning_rel: bool = False,
            lr: float = 3e-4,
            betas=(0.9, 0.95),
            weight_decay: float = 1e-4,
            warmup_steps: int = 2000,
            min_lr: float = 1e-6,
            grad_clip: float = 1.0,
            flow_steps: int = 100000,
            ema_decay: float = 0.999,
            t_dist: str = "uniform",
            num_steps: int = 20,
            tau_star: float = 0.8,
            sample_eps_max: float = 0.0,
            sample_T: float = 1.0,
            n_action_samples: int = 1,
            val_every: int = 5000,
            save_every: int = 20000,
            log_every: int = 500,
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)
        # spawn で評価プロセスへ pickle されるので、DataLoader と生データは持たない
        del self.train_dataloader, self.test_dataloader

        self.To, self.H, self.Ta = obs_history, horizon, exec_horizon
        self.yaw_cols = list(yaw_cols or [])
        self.drop_cols = list(drop_cols or [])
        self.aligning_rel = aligning_rel
        self.batch_size = train_batch_size
        self.lr, self.betas, self.weight_decay = lr, tuple(betas), weight_decay
        self.warmup_steps, self.min_lr, self.grad_clip = warmup_steps, min_lr, grad_clip
        self.flow_steps, self.ema_decay = flow_steps, ema_decay
        self.t_dist = t_dist
        self.num_steps, self.tau_star, self.sample_eps_max = num_steps, tau_star, sample_eps_max
        self.sample_T = sample_T
        self.n_action_samples = int(n_action_samples or 1)
        self.val_every, self.save_every, self.log_every = val_every, save_every, log_every
        # 学習バッチの抽出順は run.py の seed に連動させる (torch.manual_seed(seed) 直後なので
        # initial_seed() = seed)。seed 0 ではこれまでの固定値 0 と同じ。
        self.data_seed = torch.initial_seed()

        self.eval_model_name = "last_em.pth"          # EMA 重み (推論用)
        self.last_model_name = "last_em_raw.pth"      # 生重み

        # ---- 正規化統計: 向き修正後の特徴で、有効 step のみから計算 ----
        obs = self._feat(self.trainset.get_all_observations().double())
        act = self.trainset.get_all_actions().double()
        self.x_mean = obs.mean(0).float().to(device)
        self.x_std = (obs.std(0) + 1e-8).float().to(device)
        self.y_mean = act.mean(0).float().to(device)
        self.y_std = (act.std(0) + 1e-8).float().to(device)
        self.a_min = act.min(0).values.float().to(device)
        self.a_max = act.max(0).values.float().to(device)
        log.info(f"[em] feat_dim={obs.shape[-1]} yaw_cols={self.yaw_cols} drop_cols={self.drop_cols} "
                 f"aligning_rel={self.aligning_rel} "
                 f"max|z|={((obs - obs.mean(0)) / obs.std(0)).abs().max(0).values.numpy().round(1)}")

        self.obs_hist = deque(maxlen=self.To)
        self.act_queue = []

    # ------------------------------------------------------------------ data
    def _feat(self, obs):
        return build_state_features(obs, self.yaw_cols, self.drop_cols, self.aligning_rel)

    def _norm_obs(self, obs):
        return (self._feat(obs.to(self.device)) - self.x_mean) / self.x_std

    def _build_chunks(self, ds):
        """(demo, t) ごとに obs[t-To+1..t] (先頭は複製で埋める) と a[t..t+H-1]
        (エピソード末尾以降は生の 0 = 静止) を引けるよう GPU 上に正規化済みで置く。"""
        O = self._norm_obs(ds.observations)                             # (N, T, F)
        A = torch.cat([ds.actions, torch.zeros(ds.actions.shape[0], self.H, ds.actions.shape[2])], 1)
        A = (A.to(self.device) - self.y_mean) / self.y_std              # (N, T+H, 2)
        L = ds.masks.sum(1).long()
        idx = torch.cat([torch.stack([torch.full((int(l),), i), torch.arange(int(l))], 1)
                         for i, l in enumerate(L)]).to(self.device)     # (M, 2)
        return O, A, idx

    def _batch(self, data, rows):
        O, A, idx = data
        i, t = idx[rows, 0], idx[rows, 1]
        to = (t[:, None] + torch.arange(-self.To + 1, 1, device=self.device)).clamp(min=0)
        ta = t[:, None] + torch.arange(self.H, device=self.device)
        return O[i[:, None], to], A[i[:, None], ta]                     # (B,To,F), (B,H,2)

    # ------------------------------------------------------------------ losses
    def _flow_loss(self, c, a):
        x0 = torch.randn_like(a)
        if self.t_dist == "beta":
            t = torch.distributions.Beta(1.5, 1.0).sample((a.shape[0], 1, 1)).to(a.device)
        else:
            t = torch.rand(a.shape[0], 1, 1, device=a.device)
        x_t = (1 - t) * x0 + t * a
        v = self.model.velocity(x_t, c, create_graph=True)
        return ((v - (a - x0)) ** 2).mean()

    def _eps_at(self, t, eps_max):
        """サンプラのノイズ ε(t) (原論文): t < tau_star は 0、tau_star..1 で線形に eps_max まで、t >= 1 は eps_max。"""
        if t < self.tau_star:
            return 0.0
        if t < 1.0:
            return eps_max * (t - self.tau_star) / (1.0 - self.tau_star)
        return eps_max

    def _lr_at(self, step, total):
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        p = min(max((step - self.warmup_steps) / max(1, total - self.warmup_steps), 0.0), 1.0)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + math.cos(math.pi * p))

    # ------------------------------------------------------------------ training
    def train_agent(self):
        train = self._build_chunks(self.trainset)
        val = self._build_chunks(self.valset)
        log.info(f"[em] train pairs {len(train[2])} / val pairs {len(val[2])}")
        self._run_flow(train, val)
        log.info("Training done!")

    def _run_flow(self, train, val):
        total = self.flow_steps
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=self.betas,
                                weight_decay=self.weight_decay)
        if 1.0 / (1.0 - self.ema_decay) >= 0.5 * total:
            log.warning(f"[em] EMA 窓 {1/(1-self.ema_decay):.0f} step が学習の {total} step の半分以上")
        ema = _EMA(self.model, self.ema_decay)
        M = len(train[2])
        g = torch.Generator(device=self.device).manual_seed(self.data_seed)
        for step in tqdm(range(total), desc="flow", mininterval=30):
            for pg in opt.param_groups:
                pg["lr"] = self._lr_at(step, total)
            self.model.train()
            obs, a = self._batch(train, torch.randint(0, M, (self.batch_size,), device=self.device, generator=g))
            c = self.model.encode(obs)
            loss = self._flow_loss(c, a)
            rec = {"flow": loss.item()}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()
            ema.update(self.model)

            if (step + 1) % self.val_every == 0:
                rec["val_flow_ema"] = self._val_flow(val, ema.shadow)
                rec["val_flow_raw"] = self._val_flow(val, None)
            if step % self.log_every == 0 or "val_flow_ema" in rec:
                log.info(f"[flow] step {step} " + " ".join(f"{k} {v:.4f}" for k, v in rec.items()))
                wandb.log(rec)
            if (step + 1) % self.save_every == 0:
                self._save(ema, f"flow_step{step+1}")
        self._save(ema, None)

    @torch.no_grad()
    def _val_flow(self, val, weights, n_batches=8):
        """記録のみ (ckpt 選択には使わない)。seed 固定で測定ノイズを消す。"""
        backup = None
        if weights is not None:
            backup = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(weights)
        self.model.eval()
        g = torch.Generator(device=self.device).manual_seed(123)
        st = torch.cuda.get_rng_state(self.device) if self.device.startswith("cuda") else None
        torch.manual_seed(123)
        tot = 0.0
        for _ in range(n_batches):
            obs, a = self._batch(val, torch.randint(0, len(val[2]), (1024,), device=self.device, generator=g))
            with torch.enable_grad():
                tot += self._flow_loss(self.model.encode(obs), a).item()
        if st is not None:
            torch.cuda.set_rng_state(st, self.device)
        if backup is not None:
            self.model.load_state_dict(backup)
        return tot / n_batches

    def _save(self, ema, tag):
        ema_name = self.eval_model_name if tag is None else f"em_{tag}.pth"
        raw_name = self.last_model_name if tag is None else f"em_{tag}_raw.pth"
        torch.save(ema.shadow, os.path.join(self.working_dir, ema_name))
        torch.save(self.model.state_dict(), os.path.join(self.working_dir, raw_name))

    # BaseAgent の抽象メソッド (D3IL の学習ループは使わない)
    def train_step(self, state, action):
        raise NotImplementedError

    def evaluate(self, state, action):
        raise NotImplementedError

    # ------------------------------------------------------------------ inference
    def reset(self):
        self.obs_hist.clear()
        self.act_queue = []

    @torch.no_grad()
    def _integrate(self, c):
        # sample_T > 1 なら t = 1 の先も同じ dt で積分する (原論文の sampling time τ_s)
        dt = 1.0 / self.num_steps
        x = torch.randn(c.shape[0], self.H, self.model.action_dim, device=c.device)
        for i in range(int(round(self.sample_T * self.num_steps))):
            x = x + self.model.velocity(x, c) * dt
            eps = self._eps_at(i * dt, self.sample_eps_max)
            if eps > 0:
                x = x + (2 * eps * dt) ** 0.5 * torch.randn_like(x)
        return x

    @torch.no_grad()
    def _sample(self, c):
        """c: (B, cond) -> (B, H, A) 正規化済みチャンク。
        n_action_samples=K>1 なら観測ごとに K 本を同時に積分し、V(x|c) が最小のものを返す。
        比べるのは同じ c の中だけなので、Flow 学習で V の定数項が c ごとに不定でも構わない。
        K=1 は従来の _integrate と同一 (乱数の消費も同じ)。"""
        K = self.n_action_samples
        if K <= 1:
            return self._integrate(c)
        B = c.shape[0]
        c_rep = c.repeat_interleave(K, dim=0)
        x = self._integrate(c_rep)
        V = self.model.potential(x, c_rep).reshape(B, K)
        x = x.reshape(B, K, self.H, -1)
        return x[torch.arange(B, device=x.device), V.argmin(1)]

    @torch.no_grad()
    def predict(self, state, if_vision=False):
        """state: 生の観測 (obs_dim,) -> 次の 1 step の行動 (1, action_dim) (生スケール)。"""
        f = self._norm_obs(torch.as_tensor(state, dtype=torch.float32)[None])[0]
        if len(self.obs_hist) == 0:
            for _ in range(self.To):
                self.obs_hist.append(f)
        else:
            self.obs_hist.append(f)
        if not self.act_queue:
            self.model.eval()
            c = self.model.encode(torch.stack(tuple(self.obs_hist))[None])
            a = self._sample(c)[0] * self.y_std + self.y_mean                 # (H, 2)
            a = torch.max(torch.min(a, self.a_max), self.a_min)
            self.act_queue = list(a[: self.Ta].cpu().numpy())
        return self.act_queue.pop(0)[None]
