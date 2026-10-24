"""Energy Matching agent for D3IL (image input: bp-cam + inhand-cam, action chunking).

EMAgent (状態版) の学習ループ・損失・サンプラをそのまま使い、観測だけを画像に替える。
- 画像は全デモ分を uint8 で 1 本の flat tensor にまとめ、(demo, t) から直接バッチを切り出す。
  img_gpu_max_gb 以下なら GPU に置き (Aligning 900 demo で約 10GB)、超える場合
  (Sorting6 / Stacking) は CPU の pinned memory に置いて必要なフレームだけ転送する。
- 入力は D3IL の vision ベースラインと同じく画像のみ (robot_ee_pos は使わない)。
- 評価プロセスへ spawn で pickle されるので、画像ストアは __getstate__ で落とす。
"""
import logging
from collections import deque

import torch
from omegaconf import DictConfig

from agents.em_agent import EMAgent

log = logging.getLogger(__name__)


def _to_uint8(frames):
    """list of (n_i, 3, H, W) uint8 または float [0,1] -> flat (sum n_i, 3, H, W) uint8, offsets (N,), n (N,)"""
    n = torch.tensor([len(f) for f in frames])
    off = torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)[:-1]])
    flat = torch.cat([f if f.dtype == torch.uint8 else (f * 255).round().to(torch.uint8) for f in frames])
    return flat, off, n


class EMVisionAgent(EMAgent):

    def __init__(self, model: DictConfig, trainset: DictConfig, valset: DictConfig,
                 img_gpu_max_gb: float = 16.0, **kwargs):
        super().__init__(model, trainset, valset, **kwargs)
        self.img_gpu_max_gb = img_gpu_max_gb
        # 生の float 画像 (Aligning で 40GB 超) は uint8 に詰め直して手放す
        self._stores = {name: self._compact(ds) for name, ds in
                        (("train", self.trainset), ("val", self.valset))}
        del self.trainset, self.valset
        self.img_hist = deque(maxlen=self.To)

    def _compact(self, ds):
        bp, off, n = _to_uint8(ds.bp_cam_imgs)
        ih, off2, _ = _to_uint8(ds.inhand_cam_imgs)
        assert torch.equal(off, off2)
        A = torch.cat([ds.actions, torch.zeros(ds.actions.shape[0], self.H, ds.actions.shape[2])], 1)
        # フレーム数が状態より少ないデモでは、隣のデモの画像を読まないよう有効長を切り詰める
        L = torch.minimum(ds.masks.sum(1).long(), n)
        return dict(bp=bp, ih=ih, off=off, A=A, L=L)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_stores", None)
        return state

    # ------------------------------------------------------------------ data
    def _build_chunks(self, store):
        A = (store["A"].to(self.device) - self.y_mean) / self.y_std          # (N, T+H, 3)
        idx = torch.cat([torch.stack([torch.full((int(l),), i), torch.arange(int(l))], 1)
                         for i, l in enumerate(store["L"])]).to(self.device)  # (M, 2)
        gb = (store["bp"].numel() + store["ih"].numel()) / 1e9
        if gb <= self.img_gpu_max_gb:
            imgs = (store["bp"].to(self.device), store["ih"].to(self.device), store["off"].to(self.device))
        else:
            log.info(f"[em-vision] 画像 {gb:.1f}GB > {self.img_gpu_max_gb}GB: CPU pinned memory に置く")
            imgs = (store["bp"].pin_memory(), store["ih"].pin_memory(), store["off"])
        return imgs, A, idx

    def _batch(self, data, rows):
        (bp, ih, off), A, idx = data
        i, t = idx[rows, 0], idx[rows, 1]
        to = (t[:, None] + torch.arange(-self.To + 1, 1, device=self.device)).clamp(min=0)
        ta = t[:, None] + torch.arange(self.H, device=self.device)
        flat = off.to(i.device)[i][:, None] + to                              # (B, To)
        if bp.device != flat.device:                                          # CPU に置いた場合
            flat = flat.cpu()
            bp, ih = (x[flat].pin_memory().to(self.device, non_blocking=True) for x in (bp, ih))
        else:
            bp, ih = bp[flat], ih[flat]
        obs = (bp.float() / 255.0, ih.float() / 255.0)                        # (B,To,3,H,W) x2
        return obs, A[i[:, None], ta]

    # ------------------------------------------------------------------ training
    def train_agent(self):
        train = self._build_chunks(self._stores["train"])
        val = self._build_chunks(self._stores["val"])
        log.info(f"[em-vision] train pairs {len(train[2])} / val pairs {len(val[2])} "
                 f"/ frames {len(train[0][0])}")
        self._run_flow(train, val)
        del train, val
        self._stores = None
        torch.cuda.empty_cache()
        log.info("Training done!")

    # ------------------------------------------------------------------ inference
    def reset(self):
        self.img_hist.clear()
        self.act_queue = []

    @torch.no_grad()
    def predict(self, state, if_vision=True):
        """state: (bp (3,H,W), inhand (3,H,W) float [0,1], des_robot_pos) -> (1, action_dim)"""
        bp, ih, _ = state
        frame = (torch.as_tensor(bp, dtype=torch.float32, device=self.device),
                 torch.as_tensor(ih, dtype=torch.float32, device=self.device))
        if len(self.img_hist) == 0:
            for _ in range(self.To):
                self.img_hist.append(frame)
        else:
            self.img_hist.append(frame)
        if not self.act_queue:
            self.model.eval()
            obs = tuple(torch.stack([f[k] for f in self.img_hist])[None] for k in range(2))
            c = self.model.encode(obs)
            a = self._sample(c)[0] * self.y_std + self.y_mean
            a = torch.max(torch.min(a, self.a_max), self.a_min)
            self.act_queue = list(a[: self.Ta].cpu().numpy())
        return self.act_queue.pop(0)[None]
