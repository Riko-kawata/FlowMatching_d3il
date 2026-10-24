# hiruma Energy Matching (EM) モデルのパラメータ構成 (2026-10-20 時点)

現行のベースラインは `em_runs/em_hiruma_v4_play_randlean` (以下 v4)。
free_energy_* / fe_* / vis_fe_* の各スクリプトはすべてこのランの `final.pt` を既定で読む。
コードは `../Door_claude/sim/em` (train.py, model.py, dataset.py, em_core/) を無改変で使っている。
v5 (`em_dit/`, `em_unet/`) は同コードのコピーで、エネルギーヘッドだけを差し替えたもの。

学習設定 (config.json の `args` / `em`) は v1〜v5 の全ランで共通。違うのはデータセット (v1〜v4) と
エネルギーヘッドの構造 (v5)、および v5 の B 実験で振った lambda_cd / seed のみ。

---

## 1. 入出力

| 項目 | 値 | 備考 |
| --- | --- | --- |
| 制御周波数 | 10 Hz | robosuite `control_freq=10` |
| 観測ウィンドウ `obs_seq_len` | 5 フレーム | 直近 5 ステップ (0.5 s) |
| 行動チャンク `action_seq_size` | 4 ステップ | 最後の観測フレームから 4 ステップ分 |
| データ窓長 | 8 | `obs_seq_len + action_seq_size - 1` |
| 画像 | agentview 112x112 RGB uint8 | 学習時 96x96 ランダムクロップ、推論時センタークロップ、ImageNet 正規化 |
| 低次元観測 `low_dim` | 21 次元 / フレーム | joint 8 (Panda 7 関節 + gripper qpos[0]) + force 3 + ee_torque 3 + torque 7 (qfrc_actuator) |
| 行動 `action_dim` | 7 | OSC_POSE: Δpos 3 (±1 → ±5 cm) + Δrot 3 (±1 → ±0.5 rad) + gripper 1 |
| エネルギー入力 x | 4 x 7 = 28 次元 | 正規化済み行動チャンク |
| 正規化 | per-dim min-max → [-1, 1] | 学習セットで fit、`scalers.npz` に保存 (obs_/act_ の min, max, span) |
| 行動のクラスラベル | モデルに入れない | push / pull / slide はカメラと力覚から読む |

## 2. ネットワーク構成 (`model.DoorEMPolicy`, `em_core.energy_net.EnergyNet`)

```
img (B,5,3,112,112) ─ crop96 ─ ResNet18(GroupNorm, fc→64) ─┐  bf16
                                                            ├─ concat → (B,5,85) ─ ObsEncoder MLP ─ c (B,256)
low (B,5,21) ────────────────────────────────────────────────┘
x (B,4,7) ──────────────────────────────── EnergyHead(x, c) ─ V (B,)   fp32 (二階微分あり)
velocity: v = -∂V/∂x
```

### 共通部 (全ラン同一)

| モジュール | 構成 | パラメータ数 |
| --- | --- | --- |
| 画像エンコーダ | torchvision resnet18 (weights=None)、BatchNorm→GroupNorm (16 ch/group)、fc 512→`img_feat_dim`=64 | 11,209,344 |
| ObsEncoder | Linear(85·5=425→256) SiLU → Linear(256→256) SiLU → Linear(256→`cond_dim`=256) | 240,640 |

DoorEMPolicy のデフォルト引数: `img_feat_dim=64, cond_dim=256, obs_hidden_dim=256, energy_hidden_dim=512, num_hidden_layers=4, crop_size=96, use_low_dim_obs=True`。
エネルギーポテンシャル V は時間 t を入力に取らない (time-independent scalar field)。

### エネルギーヘッド (ランごとに異なる唯一の構造差)

| ラン | ヘッド | 構成 | ヘッド params | モデル合計 |
| --- | --- | --- | --- | --- |
| v1〜v4 (現行) | MLP | act_in: Linear(28→512) SiLU; net: Linear(512+256→512) SiLU → [Linear(512→512) SiLU]×3 → Linear(512→1) | 1,197,057 | 12,647,041 |
| v5 `em_dit` | DiT | トークン=行動ステップ(4)、`d_model`=256, heads 4, mlp_ratio 4, depth 4 (adaLN-Zero を c で変調、t 埋め込みなし)、学習可能 pos emb、最終 adaLN → 平均プーリング → Linear(256→1) | 4,868,865 | 16,318,849 |
| v5 `em_unet` | UNet1D | Diffusion Policy の ConditionalUnet1D 相当。down dims (128,256)、kernel 3、GroupNorm 8、Mish、FiLM を c で変調、up 後 Conv1dBlock(128) → 時間平均 → Linear(128→1) | 3,031,809 | 14,481,793 |

DiT の内訳: blocks 4,733,952 / ada_f 131,584 / act_in 2,048 / out 257。
UNet の内訳: down 1,320,960 / mid 1,052,672 / up 608,512 / final 49,536 / out 129。
DiT の attention は `F.scaled_dot_product_attention` を使わず手書き (二階微分のため)。

## 3. 学習目標

### Phase 1: Flow Matching (epoch 0〜269)

- カップリング独立 (minibatch-OT なし): x0 ~ N(0, I)、x1 = 正規化行動チャンク、t ~ U(0,1) (`t_dist="uniform"`)
- x_t = (1−t)·x0 + t·x1、目標速度 u = x1 − x0
- 損失: `mean ||(−∇_x V(x_t|c)) − u||²` (create_graph=True の二階逆伝播)

### Phase 2: Flow + Contrastive Divergence (epoch 270〜299, `phase1_epochs = round(0.9·epochs)`)

| パラメータ | 値 | 意味 |
| --- | --- | --- |
| `lambda_cd` | 1e-3 | CD 項の重み (v5 B 実験では 0 / 3e-4 も) |
| `n_gibbs` | 20 | 負例生成の Langevin ステップ数 |
| `cd_dt` | 0.01 | Langevin のステップ幅 |
| `eps_max` | 0.01 | Langevin ノイズ強度 (noise std = sqrt(2·eps·dt) ≈ 0.014) |
| `cd_trim_fraction` | 0.1 | E_neg の上位 10 % (高エネルギー外れ値) を捨てて平均 |
| `cd_neg_clamp` | 0.02 | CD 損失を −0.02 で下限クリップ |
| split negative | 固定 | バッチの半分は実データから、半分はノイズから Langevin 開始 |
| chain | detach | Langevin 連鎖は微分しない |

CD 損失 = λ·(mean E_pos − trimmed-mean E_neg)。負例は正例と同じ条件 c (detach) で生成。
phase 2 入口で `pre_cd.pt` (model/ema/opt/step) を保存し、B 実験の分岐点にしている。

## 4. 最適化・学習ハイパーパラメータ (`config.json`)

| パラメータ | 値 |
| --- | --- |
| `epochs` | 300 |
| `phase1_epochs` | 270 (自動: 0.9 × 300) |
| `batch_size` | 64 (drop_last) |
| optimizer | Adam, `lr`=1e-4, betas (0.9, 0.95), weight_decay 0 |
| `warmup_steps` | 500 (線形 lr ウォームアップ、その後一定) |
| `grad_clip` | 1.0 (global norm) |
| EMA decay | 0.999 (phase 1) → 0.99 (phase 2)。推論・保存はすべて EMA 重み |
| `seed` | 0 (torch / numpy、および val split の乱数) |
| `val_episodes` | 5 / 挙動 (= 15 episode をエピソード単位で hold-out) |
| `eval_every` | 5 epoch (seeded val flow と K=1 サンプル行動 MSE) |
| `eval_seed` | 0 |
| `num_workers` | 6 |
| 混合精度 | ResNet18 のみ bf16 autocast、エネルギーヘッドは fp32 |
| steps / epoch | 15643 // 64 = 244 → 約 73k steps / 300 epoch |

v4 の学習時間: 約 33 分 (RTX 5080)。v5 DiT: 約 82 分。

## 5. サンプラー / 推論 (`em_core/sample.py`, `eval_rollout.py`)

| パラメータ | 値 | 意味 |
| --- | --- | --- |
| `num_steps` | 20 | Euler 積分ステップ数 |
| `sample_dt` | None → 1/20 = 0.05 | 固定ステップ幅 |
| `tau_star` | 1.0 | t ≥ tau_star で Langevin ノイズ。1.0 なので **ノイズなし、決定的 ODE** |
| `eps_max` | 0.01 | (tau_star=1 のため推論では使われない) |
| `n_action_samples` K | config 上 1、ロールアウトでは `--k 3` | best-of-K: K 本サンプルし V(a|c) 最小を実行 |
| 再計画 | 4 ステップごと | チャンク全体を実行してから再サンプル |
| `horizon` | 160 ステップ (16 s) | デモは約 120〜130 ステップ |
| 成功判定 | pull +0.30 rad / push −0.30 rad / slide +0.25 m | `teach_auto.VARIANTS` |
| 評価 seed | 1000〜1049 (50 episode / 挙動) | 収集 seed と非重複 |
| 初期窓 | frame 0 を 5 枚にパディング | 学習には出ない窓 (UNet 失敗の原因) |

v5 DiT のサンプラー掃引 (steps 20/40 × K 3/5/10) では 20 steps, K=3 が最良で、既定を維持。

## 6. 学習データ (v4: `data/em_play_lean`)

| 項目 | 値 |
| --- | --- |
| 収集 | `collect.py --door-set play --lean-max 0.02 --episodes 50 --noise 0.03` (teach_auto.ScriptedDoorPolicy) |
| エピソード数 | 150 (pull / push / slide × 50、全て成功) |
| split | 135 train / 15 val、窓 15,643 train / 1,727 val |
| 1 episode 長 | 約 120〜130 ステップ @10 Hz |
| ドアセット | play (uniform 形状 + ロック中に約 2 cm の遊び、hinge 0.045 rad / slide 0.02 m) |
| 多様化 | 行動ガウスノイズ σ=0.03、ドア位置 jitter ±2 cm、robosuite 初期姿勢ノイズ、押し付け lean を半径 2 cm で毎 episode ランダム (LEAN_COMP でバイアス相殺) |
| npz キー | image, handeye, joint(8), pose(7), action(7), gripper(2), torque(7), force(3), ee_torque(3), opening, latch, phase, params |

ロボットは Panda + OSC_POSE、カメラは agentview (学習) と robot0_eye_in_hand (記録のみ、モデルは未使用)。

## 7. ラン一覧と差分

| ラン | データ / 変更点 | params | val flow (ep299) | final K=3 成功率 (push/pull/slide → mean) |
| --- | --- | --- | --- | --- |
| em_hiruma_v1 | `data/em` left ドア (push に遊び、slide は見た目が違う) | 12.65M | 0.138 | 0.94/0.90/0.70 → 0.847 |
| em_hiruma_v2_uniform | `data/em_uniform` 3 ドア見分け不可、遊びなし | 12.65M | — | 0.72/0.54/0.34 → 0.533 |
| em_hiruma_v3_play | `data/em_play` play ドア、固定 lean | 12.65M | — | 0.78/0.44/0.66 → 0.627 |
| **em_hiruma_v4_play_randlean (現行)** | `data/em_play_lean` play ドア、ランダム lean | 12.65M | 0.108 | 0.74/0.66/0.58 → 0.660 |
| em_hiruma_v5_dit | v4 データ、MLP ヘッド → DiT | 16.32M | 0.101 | 0.68/0.70/0.70 → 0.693 |
| em_hiruma_v5_dit_cd0 | DiT、pre_cd.pt から lambda_cd=0 | 16.32M | 0.096 | 0.76/0.78/0.60 → 0.713 |
| em_hiruma_v5_dit_cd3e-4 | DiT、pre_cd.pt から lambda_cd=3e-4 | 16.32M | 0.096 | 0.70/0.78/0.64 → 0.707 |
| em_hiruma_v5_dit_seed1 / seed2 | DiT、seed 1 / 2 (val split も変わる) | 16.32M | 0.100 / 0.102 | 0.707 / 0.693 |
| em_hiruma_v5_unet | v4 データ、MLP ヘッド → UNet1D | 14.48M | 0.096 | 0.02/0.04/0.02 → 0.027 (初期パディング窓で把持が閉じる) |

## 8. チェックポイント (各 run ディレクトリ)

| ファイル | 内容 |
| --- | --- |
| `pre_cd.pt` | epoch 270 の model / ema / opt / step (phase 2 分岐用、約 200 MB) |
| `best_flow.pt` | val flow 最小 epoch の EMA 重み (v4: epoch 274、DiT: epoch 274) |
| `last.pt` | 最後に評価した epoch の EMA 重み |
| `final.pt` | epoch 299 の EMA 重み。**解析スクリプトの既定** |
| `config.json` | args / em / low_dim / action_dim / params |
| `scalers.npz` | obs_min/max/span (21), act_min/max/span (7) |

v5 のチェックポイントはヘッド構造が違うため `Door_claude/sim/em/model.py` では読めない。各フォルダを sys.path に入れる必要がある。

## 9. 関連ファイル

- 学習コード: `../Door_claude/sim/em/{train.py, model.py, dataset.py}`, `em_core/{energy_net.py, flow.py, cd.py, sample.py, ema.py}`
- 起動: `run_em_hiruma.sh` (v1〜v4), `em_runs/run_v{3,4}*.sh`, `em_dit/run.sh`, `em_dit/run_B.sh`, `em_unet/run.sh`
- 評価: `eval_rollout.py` (--k, --horizon 160, --door-set play, --seed0 1000), `em_dit/eval_rollout_sampler.py`
- エネルギー / 自由エネルギー解析: `energy_rollout.py`, `free_energy_rollout.py`, `free_energy_exact.py`, `free_energy_ais.py`, `ais_convergence.py` (いずれも v4 final.pt, K=3, play が既定)
- 各ランの詳細結果: `em_runs/*/README.md`, `em_dit/README.md`, `em_unet/README.md`
