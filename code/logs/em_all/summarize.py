"""logs/em_all の評価ログ (eval_ta*.log) から成功率表を作る。Usage: python logs/em_all/summarize.py"""
import glob, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(f"{HERE}/*/*/eval_ta*.log")):
    task, run = f.split("/")[-3:-1]
    ta = int(re.search(r"eval_ta(\d+)", f).group(1))
    txt = open(f).read()
    if "RESULT" not in txt:
        continue
    vals = {k: float(v) for k, v in re.findall(r"^(Successrate(?:_\d)?|\d_box_rate|mean_inserted|KL|entropy) (\S+)$", txt, re.M)}
    n = re.search(r"n_contexts=(\d+)", txt), re.search(r"n_trajectories_per_context=(\d+)", txt)
    mod, seed = run.rsplit("_s", 1)
    rows.setdefault((task, mod, ta), []).append((int(seed), vals, int(n[0].group(1)) * int(n[1].group(1))))

print(f"{'task':10s} {'input':7s} {'Ta':>3s}  {'seed別 成功率 (%)':28s} {'平均':>6s} {'±SE':>5s}  補足")
for (task, mod, ta), rs in sorted(rows.items()):
    rs.sort()
    s = [100 * v["Successrate"] for _, v, _ in rs]
    n = sum(k for *_, k in rs)
    p = np.mean(s) / 100
    se = 100 * np.sqrt(p * (1 - p) / n)
    extra = ""
    if "Successrate_1" in rs[0][1]:
        extra = "1箱 %.1f / 2箱 %.1f" % tuple(100 * np.mean([v[k] for _, v, _ in rs]) for k in ("Successrate_1", "Successrate_2"))
    if "1_box_rate" in rs[0][1]:
        extra = "1個 %.1f / 2個 %.1f" % tuple(100 * np.mean([v[k] for _, v, _ in rs]) for k in ("1_box_rate", "2_box_rate"))
    if "KL" in rs[0][1]:
        extra += " KL %.3f" % np.mean([v["KL"] for _, v, _ in rs])
    seeds = " ".join(f"s{sd}:{100 * v['Successrate']:.1f}" for sd, v, _ in rs)
    print(f"{task:10s} {mod:7s} {ta:3d}  {seeds:28s} {np.mean(s):6.1f} {se:5.1f}  {extra} (n={n})")
