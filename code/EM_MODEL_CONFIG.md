# D3IL 用 Energy Matching (EM) モデルのパラメータ構成

**作成日**: 2026-10-20
**対象**: このディレクトリの現行既定 (`configs/agents/em_agent.yaml`, `configs/agents/em_vision_agent.yaml`)。
実装は `agents/em_agent.py` (状態入力), `agents/em_vision_agent.py` (画像入力), `agents/models/em/em_policy.py` (ネットワーク)。
実験結果は `EM_ALL_TASKS_RESULTS.md`, `SORTING4_EM_RESULTS.md`, `ALIGNING_EM_RESULTS.md` を参照。

学習設定は全タスク・全 seed で共通。タスクで違うのは観測・行動の次元 (= 最初の Linear と `in_proj` の形) と
データだけで、Phase1/Phase2 の step 数・学習率・CD の設定・サンプラはすべて同じ。

---

## 0. 一言でいうと

- **モデル**: 観測履歴 To=2 step を条件 c に埋め込み、行動チャンク x (H=16 step) のスカラーエネルギー V(x|c) を出す。
  速度場は v = −∇ₓV(x|c) で、V は時刻 t を入力に取らない (time-independent)。
- **学習**: Phase1 = Flow Matching (状態 100k step / 画像 60k step) → Phase2 = Flow + Contrastive Divergence (10k step)。
  評価は Phase2 後の EMA 重み `last_em.pth`。
- **推論**: N(0, I) から Euler 20 step の決定的 ODE で x を積分し、先頭 Ta=8 step だけ実行して観測し直す。
- **規模**: 状態入力 約 1.73M パラメータ、画像入力 約 24.0M (うち ResNet18 ×2 台が 22.4M)。

---

## 1. 入出力

| 項目 | 値 | 備考 |
|---|---|---|
| 制御周期 | 0.035 s (≈ 28.6 Hz) | MuJoCo dt 0.001 × `n_substeps` 35 (Sorting / Aligning / Inserting)。Stacking は 30 substeps = 0.030 s |
| 観測履歴 To (`obs_history`) | 2 step | エピソード先頭は最初の観測を複製して埋める |
| 予測チャンク H (`horizon`) | 16 step (≈ 0.56 s) | エピソード末尾以降の行動は 0 (静止) で埋める |
| 実行長 Ta (`exec_horizon`) | 8 step | 評価時に `agents.exec_horizon=2` で上書きすることが多い (§6) |
| データ窓長 | To + H − 1 = 17 step | (demo, t) ごとに obs[t−1..t], a[t..t+15] |
| エネルギー入力 x | H × action_dim = 32 (act=2) / 48 (act=3) / 128 (act=8) | 正規化済み行動チャンク |
| 観測の正規化 | z-score | 学習データ全有効 step の平均・標準偏差。特徴変換 (§1b) 後に計算 |
| 行動の正規化 | z-score | 推論時に逆変換し、学習データの min/max にクランプ |
| 正規化統計の保存 | **保存しない** | エージェント生成時に学習データから再計算する。ckpt 評価時もデータセットを読み直す (`analysis/eval_vision_ckpt.py`) |
| 学習ペアの取り方 | (demo, t) を一様サンプル | 有効 step のみ。乱数は `run.py` の seed に連動 |
| 行動のモード (どの箱から押すか等) | モデルに入れない | 評価時に `mode` として記録するだけ |

### 1a. タスクごとの観測・行動の中身 (状態入力)

| タスク | obs_dim | 観測の列 | action_dim | 行動 |
|---|---:|---|---:|---|
| Sorting4 | 16 | ロボット目標 xy (2), 実 xy (2), 箱 4 個 × [xy (2), tan(yaw) (1)] | 2 | 目標 xy の 1 step 差分 (Δdes_c_pos) |
| Sorting6 | 22 | 同上、箱 6 個 | 2 | 同上 |
| Inserting | 13 | ロボット目標 xy (2), 実 xy (2), 箱 3 個 × [xy, tan(yaw)] | 2 | 同上 |
| Aligning | 20 | 目標 xyz (3), 実 xyz (3), 押す箱 pos (3) + quat wxyz (4), 目標箱 pos (3) + quat (4) | 3 | 目標 xyz の差分 |
| Stacking | 20 | 目標関節角 (7), グリッパ幅 (1), 箱 3 個 × [xyz (3), tan(yaw) (1)] | 8 | 目標関節角の差分 (7) + グリッパ幅 (1) |

- 箱の向きは D3IL 標準の tan(yaw) で、yaw ≈ ±90° で発散する (Sorting4 で z-score 後 |z| ≈ 56、Stacking で 664)。
  `yaw_cols` はこれを sin4θ/cos4θ に置き換えるための対策 (§1b)。
- 画像入力の `obs_dim` (Sorting 2, Stacking 8) はロボット目標位置・関節角のみで、**モデルには入れない** (画像だけを使う)。

### 1b. 状態入力の特徴変換オプション (すべて既定 OFF)

| オプション | 既定 | 内容 | 使った実験 |
|---|---|---|---|
| `yaw_cols` | `[]` | tan(yaw) 列を [sin 4θ, cos 4θ] に置換 (列 1 本 → 2 本) | Sorting4 B_yawfix: `[6,9,12,15]` |
| `drop_cols` | `[]` | 生の観測から列を落とす (`yaw_cols` と併用不可) | Aligning B_drop: `[8,10,11,15,17,18]` (箱 z と傾きのノイズ列) |
| `aligning_rel` | `False` | Aligning 用の相対特徴 6 次元 (箱−カーソル xy, 目標−箱 xy, sin/cos Δyaw) を連結 | Aligning C_rel (B_drop と併用) |

---

## 2. ネットワーク構成 (状態入力: `EMStatePolicy`)

```
obs (B, To=2, feat_dim)  --flatten--> obs_enc (MLP) --> c (B, 256)
x   (B, H=16, action_dim) --+
                            +--> TemporalEnergyHead --> V (B,)      v = -grad_x V
c   (B, 256) ---------------+
```

全層 fp32 (autocast / bf16 は使わない)。学習時の Flow 損失は v = −∇ₓV を通るので二階微分になる。

### 2a. 観測エンコーダ `obs_enc`

| 層 | 形 |
|---|---|
| Linear | feat_dim × 2 → 512 (`obs_hidden`) |
| SiLU | |
| Linear | 512 → 512 |
| SiLU | |
| Linear | 512 → 256 (`cond_dim`) |

c は x に依存しないので、1 回の生成につき 1 回だけ計算してキャッシュする。
`EMStatePolicy` の既定引数: `obs_history=2, horizon=16, cond_dim=256, obs_hidden=512, energy_hidden=256, energy_blocks=3, energy_kernel=3`。

### 2b. エネルギーヘッド `TemporalEnergyHead` (hidden=256, 3 blocks, kernel=3, GroupNorm groups=8)

| 層 | 形 | パラメータ数 |
|---|---|---:|
| `in_proj` Linear | action_dim → 256 | 768 (act=2) |
| `pos` 学習可能な位置埋め込み | (1, 16, 256) | 4,096 |
| `cond_proj` Linear | 256 → 256、時間軸にブロードキャストして加算 | 65,792 |
| `ResBlock1d` × 3 | [GN(8) → SiLU → Conv1d(256,256,k3) → GN(8) → SiLU → Conv1d(256,256,k3)] + 残差 | 394,752 / block |
| `post` | GN(8) → SiLU → 時間軸で平均 | 512 |
| `head` | Linear 256→256 → SiLU → Linear 256→1 | 66,049 |

- 入力は h = in_proj(x) + pos + cond_proj(c) を (B, 256, 16) に転置して 1D 畳み込み。
- V は時刻 t を取らない。速度は `torch.autograd.grad` で v = −∇ₓV を計算 (学習時は `create_graph=True`)。
- robosuite_square (Door) 版の MLP ヘッドとは違い、チャンクを時間方向の系列として畳み込みで処理する。

### 2c. パラメータ数 (状態入力)

| タスク | obs_dim | feat_dim | action_dim | obs_enc | energy head | 合計 |
|---|---:|---:|---:|---:|---:|---:|
| Sorting4 (既定) | 16 | 16 | 2 | 410,880 | 1,321,473 | **1,732,353** |
| Sorting4 (yaw_cols=[6,9,12,15]) | 16 | 20 | 2 | 414,976 | 1,321,473 | 1,736,449 |
| Sorting6 | 22 | 22 | 2 | 417,024 | 1,321,473 | 1,738,497 |
| Aligning (A_raw / C_rel) | 20 | 20 | 3 | 414,976 | 1,321,729 | 1,736,705 |
| Aligning (B_drop) | 20 | 14 | 3 | 408,832 | 1,321,729 | 1,730,561 |
| Stacking | 20 | 20 | 8 | 414,976 | 1,323,009 | 1,737,985 |
| Inserting | 13 | 13 | 2 | 407,808 | 1,321,473 | 1,729,281 |

タスクによる違いは最初の Linear と `in_proj` だけで、約 76% がエネルギーヘッドにある。
`last_em.pth` は約 6.9MB (float32)。

---

## 3. 画像入力の違い (`EMVisionPolicy`, `em_vision_agent.yaml`)

```
bp-cam (B,2,3,96,96) ─ ResNet18(GN) + SpatialSoftmax ─ 64 ─┐
                                                            ├─ concat (B, 2, 128) ─ flatten ─ obs_enc ─ c (B,256)
inhand-cam (B,2,3,96,96) ─ ResNet18(GN) + SpatialSoftmax ─ 64 ─┘
x (B,16,action_dim) ─────────────────── TemporalEnergyHead(x, c) ─ V (B,)
```

| 項目 | 値 |
|---|---|
| 入力 | bp-cam と inhand-cam の 2 台、各 (3, 96, 96) RGB [0,1]、To=2 フレーム。低次元観測は使わない |
| 画像エンコーダ | D3IL vision ベースラインと同一の `MultiImageObsEncoder` |
| バックボーン | ResNet18Conv (pretrained なし、BatchNorm → GroupNorm)、カメラ間で非共有 (`share_rgb_model: False`) |
| プーリング | SpatialSoftmax 32 keypoints → 64 次元 / カメラ |
| 前処理 | `imagenet_norm: True`, `random_crop: False`, `resize_shape: null` (96×96 をそのまま入力) |
| フレーム特徴 `frame_dim` | 128 (64 × 2 台) |
| `obs_enc` | Linear 128×2=256 → 512 → SiLU → Linear 512 → 256 (状態版より 1 層少ない) |
| エネルギーヘッド | 状態版と同一 |
| パラメータ数 (act=2) | 画像エンコーダ 22,394,176 + obs_enc 262,912 + energy 1,321,473 = **23,978,561** |
| `last_em.pth` | 約 186MB |

学習の違いは `train_batch_size` 128 (状態 512)、`lr` 1.5e-4 (状態 3e-4)、`phase1_steps` 60k (状態 100k)。
画像は全デモ分を uint8 で 1 本の flat tensor にまとめ、16GB (`img_gpu_max_gb`) 以下なら GPU、超えれば CPU pinned memory に置く
(Sorting4 17.6GB / Sorting6 42.4GB / Stacking 34.1GB で、いずれも pinned memory)。

---

## 4. 学習 (Phase1: Flow Matching)

| 項目 | 状態入力 | 画像入力 |
|---|---|---|
| `phase1_steps` | 100,000 | 60,000 |
| `train_batch_size` | 512 | 128 |
| optimizer | AdamW | 同左 |
| `lr` (最大) | 3.0e-4 | 1.5e-4 |
| `betas` | (0.9, 0.95) | 同左 |
| `weight_decay` | 1.0e-4 | 同左 |
| スケジュール | 線形 warmup `warmup_steps`=2000 → cosine で `min_lr`=1e-6 まで | 同左 |
| `grad_clip` | 1.0 (global norm) | 同左 |
| `ema_decay_p1` | 0.999 (実効窓 1,000 step) | 同左 |
| `t_dist` | uniform (`beta` を選ぶと Beta(1.5, 1.0)) | 同左 |
| 混合精度 | なし (fp32) | なし |
| `seed` | config 既定 42。実験では 0 / 1 / 2 (`seed=` で上書き) | 同左 |
| val split | 別ファイル (`*_eval_files.pkl`、D3IL 標準) | 同左 |
| `num_workers` | 未使用 (DataLoader を捨て、全データを GPU に置いて直接バッチを切る) | 同左 |

Flow 損失: x₀ ~ N(0, I), t ~ U(0,1), x_t = (1−t)x₀ + t·a として ‖v(x_t, c) − (a − x₀)‖² の平均。
カップリングは独立 (minibatch-OT なし)。val flow は記録のみで、ckpt 選択には使わない (成功率と相関しないため)。

### 4a. step 数とエポック・学習時間の目安

| タスク | 学習ペア数 (demo, t) | 1 epoch 相当の step | Phase1 の epoch 換算 | 学習時間 (Phase1+2) |
|---|---:|---:|---:|---|
| Sorting4 状態 | 316,813 | 619 | 約 162 | 約 2.0 h |
| Sorting6 状態 | 765,350 | 1,495 | 約 67 | 約 3.0 h |
| Stacking 状態 | 627,453 | 1,225 | 約 82 | 約 2.1 h |
| Inserting 状態 | 692,360 | 1,352 | 約 74 | 約 1.1 h |
| Aligning 状態 | 174,574 | 341 | 約 293 | 約 2.4 h |
| Sorting4 画像 | 316,813 | 2,475 | 約 24 | 約 5.6 h |
| Sorting6 画像 | 765,350 | 5,979 | 約 10 | 約 5.6 h |
| Stacking 画像 | 616,163 | 4,814 | 約 12 | 約 4.2 h |

GPU は NVIDIA RTX 6000 Ada。学習時間は `logs/em_all` の run.log の時刻差で、1 GPU に 2 本同時に載せた状態の値。

---

## 5. 学習 (Phase2: Flow + Contrastive Divergence)

Phase1 の最終 **生** 重み (`em_p1_step*_raw.pth` = `last_em_raw.pth`) から続けて、Flow 損失 + CD 損失で `phase2_steps`=10,000 step。
既定では AdamW と EMA を作り直し、学習率も warmup から再スタートする (= Phase1 と同じスケジュールを 10k step に圧縮)。
Phase1 の optimizer 状態は保存していない。

| 項目 | 既定 | 内容 |
|---|---|---|
| `lambda_cd` | 1.0e-3 | CD 項の重み。0 で Phase2 は Flow のみ |
| `n_gibbs` | 50 | 負例の鎖の長さ (× `cd_dt`=0.01 → 時間 0.5) |
| `cd_dt` | 0.01 | 鎖の Euler 刻み |
| `cd_eps_max` | 0.1 | 鎖に足すノイズ ε (noise std = √(2·ε·dt) ≈ 0.045)。`cd_schedule=False` では最初から一定 |
| `cd_schedule` | False | True なら原論文の ε(t) (t < `tau_star` は 0、以降 `cd_eps_max` まで線形) |
| `cd_trim_fraction` | 0.1 | 負例エネルギーの上位 10% を平均から除く |
| `cd_neg_clamp` | 0.02 | CD 損失の下限 −0.02 |
| split negative | 固定 | バッチの半分はデータ、半分は N(0, I) から鎖を開始 |
| chain | detach | 鎖は微分しない。c も detach して obs_enc には CD の勾配を流さない |
| `ema_decay_p2` | 0.999 | Phase2 の EMA (Phase2 開始時のモデル重みから取り直す) |
| `phase2_lr` | null | 指定すると LR を一定 (warmup `phase2_warmup`=100) にし、EMA を `init_ema_ckpt` から引き継ぐ |
| `init_ckpt` / `init_ema_ckpt` | null | `phase1_steps=0` と併用して Phase1 の ckpt から Phase2 だけ回す |

CD 損失 = clamp((mean E_pos − trimmed mean E_neg) × λ, min=−0.02)。

> 既定の Phase2 は成功率を上げない (EM_ALL_TASKS_RESULTS.md §6)。原論文に寄せた設定
> (`n_gibbs=200, cd_schedule=true, tau_star=1.0, cd_eps_max=0.01, phase2_lr=3e-5, ema_decay_p2=0.99, phase2_steps=2000`)
> は Inserting 状態 Ta=8 でだけ +4.8pt、Stacking 画像では −17pt (同 §7)。

---

## 6. 推論 (サンプラ) と評価

### 6a. サンプラ

| 項目 | 既定 | 内容 |
|---|---|---|
| 初期値 | x ~ N(0, I) (H=16 × action_dim) | |
| `num_steps` | 20 | Euler、dt = 1/20 = 0.05 |
| `sample_T` | 1.0 | t = 1 で止める。>1 なら同じ dt でその先も積分 |
| `sample_eps_max` | 0.0 | 0 なので決定的 ODE (確率性は初期ノイズのみ)。>0 で t ≥ `tau_star` から Langevin ノイズ |
| `tau_star` | 0.8 | ノイズ ε(t) の立ち上がり時刻 (ノイズ 0 のときは無関係) |
| `n_action_samples` | 1 | K>1 で K 本積分して V(x|c) 最小を実行 (best-of-K) |
| 後処理 | 逆正規化 → 学習データの [min, max] にクランプ | |
| 再計画 | Ta step ごと | 先頭 Ta step を実行してから再サンプル。残りの H−Ta step は捨てる |
| 初期窓 | 最初の観測を To=2 枚に複製 | 学習では t=0 の窓も同じ複製で作っているので、学習・推論で一致する |

best-of-K (K=2〜16) と Langevin tail (ε=0.1 / 0.3) はどちらも効果が無かったか悪化した (SORTING4_EM_RESULTS.md §9)。
t = 2 までの SDE 積分も全条件で悪化 (EM_ALL_TASKS_RESULTS.md §7)。Heun / SDE / 積分時間の掃引は `analysis/EM_SAMPLER_ABLATION.md`。
既定は決定的 ODE のまま。

### 6b. 評価条件 (D3IL 標準 `*_Sim.test_agent`)

| タスク | 最大 step / ep | 成功判定 | 評価 (context × 回, `logs/em_all`) | 既定 config の評価 |
|---|---:|---|---|---|
| Sorting4 | 700 | 箱 4 個すべてが色ごとのゾーン内 | 60 × 18 = 1080 ep (画像 60 × 6) | 30 × 12 |
| Sorting6 | 1200 | 箱 6 個すべてが色ごとのゾーン内 | 60 × 20 = 1200 ep (画像 60 × 6) | 30 × 12 |
| Stacking | 1000 (画像 1200) | 3 箱すべてが目標箱の xy ± 0.06 内、かつ箱同士の最小 z 差 > 0.03 (積まれている)。1 箱 / 2 箱までも記録 | 60 × 18 (画像 60 × 6) | 5 × 1 |
| Inserting | 2000 | 3 箱すべてが目標との距離 ≤ 0.01。入った個数も記録 | 60 × 6 = 360 ep | 60 × 6 |
| Aligning | 400 | 位置差 ≤ 0.018 かつ 回転差 ≤ 0.048 | 60 × 8 = 480 ep | 60 × 8 |

- context は `environments/dataset/data/<task>/*_test_contexts.pkl` の固定初期配置。
- 評価の乱数 (初期ノイズ) はワーカー番号 (pid) で seed を固定するので、同じ評価条件なら run 間で同じ乱数列になる
  (Phase1/Phase2 のペア比較に利用)。
- `mode` (どの箱から片付けたか等) も記録し、D3IL の KL / entropy 指標を出す。

---

## 7. 学習データ

| タスク | 学習デモ数 | 評価デモ数 | 学習ペア数 (demo, t) | `max_len_data` | 備考 |
|---|---:|---:|---:|---:|---|
| Sorting4 | 949 | 105 | 316,813 | 600 | 箱 4 個 (赤 2 / 青 2)、1 デモ平均約 334 step |
| Sorting6 | 1,495 | 166 | 765,350 | 1200 | 箱 6 個 (赤 3 / 青 3) |
| Stacking | 1,000 | 95 | 627,453 | 1000 | 画像版は 983 / 92 デモ (616,163 ペア) |
| Inserting | 700 | 100 | 692,360 | 2400 | 1 デモ平均約 989 step |
| Aligning | 900 | 100 | 174,574 | 512 | 1 デモ平均約 194 step |

- データは D3IL 公開データセット (`environments/dataset/data/<task>/`、nakajo のコピーへの symlink)。
  学習 / 評価の split は `*_train_files.pkl` / `*_eval_files.pkl` で固定 (乱数で分けない)。
- 画像は bp-cam と inhand-cam の 96×96 RGB。Inserting には画像データが無い。
- 行動はデモ中のロボット目標位置の 1 step 差分 (§1a)。クラスラベルや言語条件は無い。

---

## 8. チェックポイントとログ (各 run ディレクトリ)

| ファイル | 内容 |
|---|---|
| `last_em.pth` | 最終 EMA 重み。**評価に使う** |
| `last_em_raw.pth` | 最終生重み (Phase2 の起点に使う) |
| `em_p1_step100000.pth` / `_raw.pth` (画像は `step60000`) | Phase1 終了時の EMA / 生重み。Phase2 の `pre_cd` 相当 |
| `em_p2_step10000.pth` / `_raw.pth` | Phase2 終了時 (= `last_em*.pth` と同じ) |
| `em_p{1,2}_step{N}.pth` | `save_every`=20,000 step ごとの途中保存 |
| `.hydra/config.yaml` | 実際に使った全設定 (再評価時はここから agent を再生成) |
| `run.log` | 学習ログ (`log_every`=1,000 step、`val_every`=5,000 step の val flow) |
| `eval_ta{8,2}.log`, `pair_{p1,p2}_ta{8,2}.{log,npz}` | 評価ログとエピソードごとの成否 (`EVAL_NPZ` で保存) |

正規化統計 (`x_mean/x_std/y_mean/y_std/a_min/a_max`) は ckpt に入っていないので、評価時はデータセットの再読み込みが必要。

---

## 9. ラン一覧と差分

| ラン (ディレクトリ) | 内容 | 主な結果 |
|---|---|---|
| `logs/sorting/em_analysis/A_raw`, `B_yawfix` | Sorting4 状態、観測そのまま vs `yaw_cols=[6,9,12,15]`。Phase1 のみ (100k step) | A: 31.2% (Ta=8)。詳細 SORTING4_EM_RESULTS.md |
| `logs/aligning/em_state/{A_raw,B_drop,C_rel}_seed{0,1}` | Aligning 状態、観測の 3 通り。Phase1 のみ (100k step) | A_raw 89.5%、B/C も数 pt 差。ALIGNING_EM_RESULTS.md |
| `logs/aligning/em_vision/seed{0,1}` | Aligning 画像、Phase1 のみ (60k step, `phase2_steps=0`) | 41.3% (2 seed) |
| `logs/em_all/<task>/state_s{0,1,2}` | Sorting4 / Sorting6 / Stacking / Inserting 状態、既定設定 | Ta=2 で 42.7 / 0.4 / 4.8 / 59.6% |
| `logs/em_all/<task>/vision_s{0,1,2}` | Sorting4 / Sorting6 / Stacking 画像 | Ta=2 で 74.4 / 73.0 / 38.1% (3 seed, Phase2) |
| `logs/em_all/p2paper/` | 原論文寄せの CD (Inserting 状態, Stacking 画像) | Inserting Ta=8 +4.8pt、Stacking 画像 −17pt |
| `logs/em_sampler_ablation_20261007/` | 推論アブレーション (Heun / SDE / 積分時間) | `analysis/EM_SAMPLER_ABLATION.md` |

すべてのランで §2〜§5 の設定は同じ。違うのはタスク・観測の特徴変換・seed と、Phase2 を回したか (`phase2_steps` は 2026-10-01 の
`logs/em_all` から既定 10k。それ以前の Sorting4 / Aligning のランは `phase2_steps=0`)、および p2paper の CD 設定だけ。

---

## 10. 関連ファイル

- **モデル / 学習**: `agents/em_agent.py`, `agents/em_vision_agent.py`, `agents/models/em/em_policy.py`, `agents/models/em/sampling_ablation.py`
- **設定**: `configs/agents/em_agent.yaml`, `configs/agents/em_vision_agent.yaml`, タスク設定 `configs/<task>_config.yaml`
- **学習の起動**: `run.py` (hydra)。一括投入は `logs/em_all/orchestrate.py` (既定), `orchestrate_p1p2.py` (Phase1/2 ペア評価), `orchestrate_p2paper.py` (原論文寄せ CD)
- **評価**: `analysis/eval_vision_ckpt.py <run_dir> <ckpt> [n_cores] [key=value ...]` (状態 / 画像どちらの ckpt も可)
- **集計**: `logs/em_all/summarize.py`, `summarize_p1p2.py`, `summarize_p2paper.py` (出力 `logs/em_all/summary_p1p2.md` など)
- **解析**: `analysis/sorting_rollout_diag.py`, `sorting_failure_videos.py`, `sorting_energy_trace.py`, `plot_sorting_energy.py`, `em_sampler_experiments.py`
- **環境**: `simulation/<task>_sim.py` (評価ループ), `environments/d3il/envs/gym_<task>_env/` (MuJoCo 環境・成功判定), `environments/dataset/<task>_dataset.py` (観測・行動の作り方)

起動例:

```bash
python run.py --config-name=sorting_4_config agents=em_agent agent_name=em seed=0 \
  wandb.entity=none wandb.project=d3il +skip_sim=true          # 学習のみ
python analysis/eval_vision_ckpt.py <run_dir> last_em.pth 30 agents.exec_horizon=2   # 評価
```

パラメータ数は `EMStatePolicy(obs_dim, action_dim, obs_history=2, horizon=16, ...)` を d3il_env で生成して数えた値。
