"""State-based Energy Matching policy for D3IL.

robosuite_square/model.py の画像版から、観測エンコーダを状態 MLP に差し替えたもの。
energy head (TemporalEnergyHead) と v = -grad_x V(x|c) の構造は同じ。

- c = obs_enc(obs history) は x_t に依存しないので 1 回だけ計算してキャッシュする。
- V は時刻 t を入力に取らない (time-independent scalar field)。
"""
import numpy as np
import torch
import torch.nn as nn


def fix_yaw_features(obs, yaw_cols):
    """tan(yaw) 列を [sin 4θ, cos 4θ] に置き換える (θ = arctan(tan yaw))。

    D3IL Sorting は箱の向きを tan(yaw) で観測に入れているが、yaw ≈ ±90° で
    tan が 1e16 まで発散し、z-score 後は 97% のサンプルで ≈0・残りが |z|≈80 の
    スパイクになる。θ = arctan(tan) は yaw mod 180° を正しく復元でき、箱は立方体
    (90° 対称) なので 4θ の sin/cos が連続・有界・対称性不変な表現になる。
    obs: (..., D) tensor -> (..., D + len(yaw_cols))
    """
    if not yaw_cols:
        return obs
    parts, prev = [], 0
    for c in yaw_cols:
        parts.append(obs[..., prev:c])
        th = torch.atan(obs[..., c:c + 1])
        parts += [torch.sin(4 * th), torch.cos(4 * th)]
        prev = c + 1
    parts.append(obs[..., prev:])
    return torch.cat(parts, dim=-1)


def aligning_rel_features(obs):
    """Aligning (obs_dim 20) 用の相対特徴 6 次元。
    列: des(0:3) c_pos(3:6) box_pos(6:9) box_quat wxyz(9:13) target_pos(13:16) target_quat(16:20)
    -> [box_xy - c_xy, target_xy - box_xy, sin/cos(target_yaw - box_yaw)]"""
    box_yaw = 2 * torch.atan2(obs[..., 12], obs[..., 9])
    tgt_yaw = 2 * torch.atan2(obs[..., 19], obs[..., 16])
    d = tgt_yaw - box_yaw
    return torch.cat([obs[..., 6:8] - obs[..., 3:5], obs[..., 13:15] - obs[..., 6:8],
                      torch.sin(d)[..., None], torch.cos(d)[..., None]], dim=-1)


def build_state_features(obs, yaw_cols=(), drop_cols=(), aligning_rel=False):
    """相対特徴 (生の列から計算) -> drop_cols を除去 -> tan(yaw) 修正 -> 相対特徴を連結。
    drop_cols は生の観測の列番号。yaw_cols と併用しない。"""
    assert not (yaw_cols and drop_cols), "yaw_cols と drop_cols の併用は未対応"
    rel = aligning_rel_features(obs) if aligning_rel else None
    if drop_cols:
        keep = [i for i in range(obs.shape[-1]) if i not in set(drop_cols)]
        obs = obs[..., keep]
    obs = fix_yaw_features(obs, yaw_cols)
    return obs if rel is None else torch.cat([obs, rel], dim=-1)


class ResBlock1d(nn.Module):
    def __init__(self, ch, kernel=3, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups, ch), nn.SiLU(),
            nn.Conv1d(ch, ch, kernel, padding=kernel // 2),
            nn.GroupNorm(groups, ch), nn.SiLU(),
            nn.Conv1d(ch, ch, kernel, padding=kernel // 2),
        )

    def forward(self, h):
        return h + self.net(h)


class TemporalEnergyHead(nn.Module):
    """robosuite_square/model.py と同じ。チャンクを時間方向の系列として扱う。"""

    def __init__(self, act_dim, horizon, cond_dim, hidden=256, n_blocks=3, kernel=3, groups=8):
        super().__init__()
        self.in_proj = nn.Linear(act_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, horizon, hidden))
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock1d(hidden, kernel, groups) for _ in range(n_blocks)])
        self.post = nn.Sequential(nn.GroupNorm(groups, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, c):
        h = self.in_proj(x) + self.pos + self.cond_proj(c).unsqueeze(1)   # (B,H,hidden)
        h = h.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        h = self.post(h).mean(-1)
        return self.head(h).squeeze(-1)                                   # (B,)


class EMStatePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, obs_history=2, horizon=16, yaw_cols=(),
                 cond_dim=256, obs_hidden=512, energy_hidden=256, energy_blocks=3,
                 energy_kernel=3, drop_cols=(), aligning_rel=False):
        super().__init__()
        self.yaw_cols = list(yaw_cols or [])
        self.feat_dim = build_state_features(torch.zeros(1, obs_dim), self.yaw_cols,
                                             list(drop_cols or []), aligning_rel).shape[-1]
        self.obs_history = obs_history
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_enc = nn.Sequential(
            nn.Linear(self.feat_dim * obs_history, obs_hidden), nn.SiLU(),
            nn.Linear(obs_hidden, obs_hidden), nn.SiLU(),
            nn.Linear(obs_hidden, cond_dim),
        )
        self.energy = TemporalEnergyHead(action_dim, horizon, cond_dim, energy_hidden,
                                         energy_blocks, energy_kernel)
        self.visual_input = False

    def get_params(self):
        return self.parameters()

    def encode(self, obs):
        """obs: (B, To, feat_dim) 正規化済み -> c: (B, cond_dim)。生成ごとに 1 回。"""
        return self.obs_enc(obs.reshape(obs.shape[0], -1))

    def potential(self, x, c):
        return self.energy(x, c)

    def velocity(self, x, c, create_graph=False):
        """v = -grad_x V(x|c)。c は定数として扱う。"""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            V = self.potential(x, c).sum()
            g = torch.autograd.grad(V, x, create_graph=create_graph)[0]
        return -g


class EMVisionPolicy(nn.Module):
    """画像版: D3IL の vision ベースラインと同じ MultiImageObsEncoder (ResNet18+GroupNorm,
    bp-cam / inhand-cam 各 64 次元) で各フレームを埋め込み、To フレーム分を連結して
    MLP で c にする。energy head と v = -grad_x V(x|c) は状態版と同じ。"""

    def __init__(self, obs_encoder, action_dim, obs_history=2, horizon=16, frame_dim=128,
                 cond_dim=256, obs_hidden=512, energy_hidden=256, energy_blocks=3,
                 energy_kernel=3, device="cpu"):
        super().__init__()
        import hydra
        self.obs_history = obs_history
        self.horizon = horizon
        self.action_dim = action_dim
        self.img_enc = hydra.utils.instantiate(obs_encoder)
        self.obs_enc = nn.Sequential(
            nn.Linear(frame_dim * obs_history, obs_hidden), nn.SiLU(),
            nn.Linear(obs_hidden, cond_dim),
        )
        self.energy = TemporalEnergyHead(action_dim, horizon, cond_dim, energy_hidden,
                                         energy_blocks, energy_kernel)
        self.visual_input = True

    def get_params(self):
        return self.parameters()

    def encode(self, obs):
        """obs: (bp, inhand) 各 (B, To, 3, 96, 96) float [0,1] -> c: (B, cond_dim)。"""
        bp, ih = obs
        B, To = bp.shape[:2]
        f = self.img_enc({"agentview_image": bp.flatten(0, 1), "in_hand_image": ih.flatten(0, 1)})
        return self.obs_enc(f.reshape(B, -1))

    def potential(self, x, c):
        return self.energy(x, c)

    def velocity(self, x, c, create_graph=False):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            V = self.potential(x, c).sum()
            g = torch.autograd.grad(V, x, create_graph=create_graph)[0]
        return -g
