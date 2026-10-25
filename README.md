# D3IL × Flow Matching — Flow 学習の結果まとめ

`experiments/d3il` で学習した方策のうち、**Flow 学習 (flow matching で v = -∇V を回帰) の重み**について、
推論動画と、学習・推論に使ったファイルをまとめたもの。

- 学習: 2026-10-01 〜 10-04、`d3il/logs/em_all/` (状態入力 100k step、画像入力 60k step)
- 成功率: [code/logs/em_all/summary_flow.md](code/logs/em_all/summary_flow.md)


## フォルダ構成

```
dl_kouza/
├── README.md
├── FLOW_D3IL_REPORT.md    実行内容・結果・工夫のまとめ
├── videos/                    推論動画 (mp4)
│   ├── <task>_<state|vision>_s0/ctx<c>_k0_<success|fail>.mp4
│   ├── <task>_<state|vision>_s0/results.json   各エピソードの成否・step 数・条件
│   ├── phase1_combined.mp4    7 条件から 1 本ずつ選んでつないだもの (順番は concat_list.txt)
│   └── logs/                  動画作成時のログ
├── runs/<task>/<state|vision>_s<seed>/   学習結果
│   ├── em_flow_step100000.pth (状態) / em_flow_step60000.pth (画像)   Flow 学習の EMA 重み
│   ├── .hydra/                学習時の設定 (config.yaml / overrides.yaml / hydra.yaml)
│   ├── run.log                学習ログ
│   ├── train.log              学習プロセスの標準出力 (ある run のみ)
│   └── flow_ta{8,2}.{log,npz} 評価ログ / エピソードごとの成否
├── code/                      学習・推論のコード (d3il と同じ相対パス)
│   ├── run.py                 学習の入口 (hydra)
│   ├── agents/em_agent.py, em_vision_agent.py, base_agent.py, models/em/em_policy.py, models/vision, ...
│   ├── configs/<task>[_vision]_config.yaml, configs/agents/em_{,vision_}agent.yaml
│   ├── simulation/*_sim.py    評価ループ
│   ├── environments/dataset/*_dataset.py
│   ├── analysis/eval_vision_ckpt.py   保存済み重みの評価
│   ├── logs/em_all/orchestrate.py         学習 + 評価の投入
│   ├── logs/em_all/orchestrate_flow.py    画像 seed 1,2 の学習と全 run の評価 (flow_ta*.npz を作る)
│   ├── logs/em_all/summarize_flow.py      成功率の集計 → summary_flow.md
│   └── EM_ALL_TASKS_RESULTS.md, EM_MODEL_CONFIG.md, EM_MODEL_PARAMS.md, requirements_d3il_env.txt
└── scripts/                   動画作成スクリプト
    ├── record_flow_videos.py
    ├── record_all.sh          全 7 条件
    └── record_vision.sh       画像入力 3 条件のみ
```

`runs/` には 21 run (状態: 4 タスク × seed 0,1,2、画像: Sorting4 / Sorting6 / Stacking × seed 0,1,2) がある。
Inserting は画像データが無いので状態入力のみ。

## 学習の設定 (code/configs/agents/em_agent.yaml)

| 項目 | 状態入力 | 画像入力 |
|---|---|---|
| `flow_steps` | 100000 | 60000 |
| `ema_decay` | 0.999 | 0.999 |
| `lr` / バッチ | 3e-4 / 512 | 1.5e-4 / 128 |
| 観測履歴 To / チャンク H | 2 / 16 | 2 / 16 |
| 推論 | Euler 20 step の決定的 ODE、best-of-K なし | 同左 |

保存される重み: `em_flow_step{N}.pth` (`save_every` ごと) と `last_em.pth` (最終、EMA)。`*_raw.pth` は EMA 前の生重み。

## 成功率 (3 seed 合算)

| タスク | 入力 | Ta=8 | Ta=2 |
|---|---|---:|---:|
| Sorting4 | 状態 | 30.9 | **46.0** |
| Sorting4 | 画像 | 60.2 | **75.8** |
| Sorting6 | 状態 | 0.2 | 0.4 |
| Sorting6 | 画像 | 47.2 | **72.3** |
| Stacking | 状態 | **7.6** | 4.2 |
| Stacking | 画像 | 35.5 | **36.8** |
| Inserting | 状態 | 52.9 | **58.9** |

± と seed 別、Stacking / Inserting の途中段階の成功率は `summary_flow.md`。

## 推論動画

- 重み: 各条件の **seed 0** の Flow 学習の重み
- テスト context 0〜4 を 1 回ずつ (各条件 5 本、計 35 本)
- 実行長 Ta: 成功率の高かった方 (Stacking 状態のみ Ta=8、他は Ta=2)
- 画面: 左 = 俯瞰カメラ `bp_cam`、右 = 手先カメラ `inhand_cam` (480×480 ずつ、H.264, 30 fps ≈ 実時間)。
  画像入力の方策が見ているのはこの 2 カメラの 96×96 画像。
- 成功したエピソードはその時点で終わるので短い。失敗は最大 step (Sorting4 700 / Sorting6 1200 / Stacking 1000〜1200 / Inserting 2000) まで続く。
- 動画内の文字は作成時の名前のまま "Phase1" と表示される (= Flow)。

| タスク | 入力 | Ta | 動画 5 本の成否 (ctx 0〜4) |
|---|---|---:|---|
| Sorting4 | 状態 | 2 | ✗ ✗ ✗ ✗ ○ (1/5) |
| Sorting4 | 画像 | 2 | ○ ✗ ✗ ✗ ○ (2/5) |
| Sorting6 | 状態 | 2 | ✗ ✗ ✗ ✗ ✗ (0/5) |
| Sorting6 | 画像 | 2 | ○ ✗ ○ ✗ ○ (3/5) |
| Stacking | 状態 | 8 | ✗ ✗ ✗ ✗ ✗ (0/5) |
| Stacking | 画像 | 2 | ✗ ✗ ○ ✗ ○ (2/5) |
| Inserting | 状態 | 2 | ○ ○ ✗ ✗ ○ (3/5) |

動画は 5 本ずつなので成功率の推定には使えない (成功率は上の表、各 1080〜3600 エピソード)。
また同じ context でも、何本目に回したか (乱数の消費) と GPU 計算の非決定性で結果が変わる。

## 再現

d3il_env (`/mnt/data1/home/rkawata/miniconda3/envs/d3il_env`) で、`experiments/` から:

```bash
bash dl_kouza/scripts/record_all.sh          # 動画 (7 条件並列)
# 1 条件だけ
MUJOCO_GL=egl python dl_kouza/scripts/record_flow_videos.py --task sorting_4 --mod vision --ta 2 \
    --seed 0 --contexts 0 1 2 3 4 --out dl_kouza/videos
# 成功率の表
python dl_kouza/code/logs/em_all/summarize_flow.py dl_kouza/runs
```

動画スクリプトは `experiments/d3il` のコードとデータ (データは symlink) で推論し、重みは `dl_kouza/runs/` から、
学習時の設定は d3il 側の同じ run の `.hydra/config.yaml` から読む。評価ループは各タスクの `Sim.test_agent` (n_cores=1)
をそのまま呼び、環境の `reset` / `step` をスクリプト内でだけ包んでカメラ画像を描画する。描画は MjData のコピーに
対して行うので推論結果は変わらない。

学習と成功率評価 (`code/` を `experiments/d3il` と同じ構成に置いた上で、そのディレクトリから):

```bash
python run.py --config-name=sorting_4_config agents=em_agent agent_name=em seed=0 \
    wandb.entity=none wandb.project=d3il +skip_sim=true
python analysis/eval_vision_ckpt.py <run_dir> em_flow_step100000.pth 30 agents.exec_horizon=2
```


