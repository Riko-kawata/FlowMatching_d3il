"""Flow 学習の重みの成功率の集計。Usage: python logs/em_all/summarize_flow.py [runs_root]

- 成功率: <runs_root>/<task>/<mod>_s<seed>/flow_ta{8,2}.npz (successes: context x trajectory)。
  runs_root の既定はこのファイルのあるディレクトリ (d3il の logs/em_all)。dl_kouza では dl_kouza/runs を渡す。
- ± は seed ごとの成功率の標準誤差ではなく、全エピソードをまとめた二項標準誤差。
"""
import glob, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
TASKS = ["sorting_4", "sorting_6", "stacking", "inserting"]


def load(run, ta):
    f = os.path.join(run, f"flow_ta{ta}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


print("## Flow 学習の成功率 (%)\n")
print("| タスク | 入力 | Ta | seed 数 | ep 数 | 成功率 | seed 別 | 補足 |")
print("|---|---|---:|---:|---:|---:|---|---|")
for task in TASKS:
    for mod in ("state", "vision"):
        for ta in (8, 2):
            succ, per_seed, extra = [], [], []
            for run in sorted(glob.glob(f"{ROOT}/{task}/{mod}_s*")):
                a = load(run, ta)
                if a is None:
                    continue
                succ.append(a["successes"].ravel())
                per_seed.append(100 * a["successes"].mean())
                if "successes_1" in a:
                    extra.append([a["successes_1"].mean(), a["successes_2"].mean()])
                if "n_inserted" in a:
                    extra.append([(a["n_inserted"] >= 1).mean(), (a["n_inserted"] >= 2).mean()])
            if not succ:
                continue
            s = np.concatenate(succ)
            p = s.mean()
            se = np.sqrt(p * (1 - p) / len(s))
            ex = ""
            if extra:
                e = 100 * np.mean(extra, 0)
                lab = ("1 箱以上", "2 箱以上") if task == "stacking" else ("1 個以上", "2 個以上")
                ex = f"{lab[0]} {e[0]:.1f}, {lab[1]} {e[1]:.1f}"
            print(f"| {task} | {mod} | {ta} | {len(per_seed)} | {len(s)} | {100 * p:.1f} ± {100 * se:.1f} "
                  f"| {', '.join(f'{x:.1f}' for x in per_seed)} | {ex} |")
