"""EM (Flow 学習: 状態 100k / 画像 60k step) を 4 タスク x 状態/画像で学習して評価する。
Usage (experiments/d3il から): nohup python logs/em_all/orchestrate.py > logs/em_all/orchestrate.out 2>&1 &
- 学習: run.py +skip_sim=true。GPU ごとに同時 MAX_TRAIN_PER_GPU 本まで。
- 評価: analysis/eval_vision_ckpt.py で last_em.pth (Flow 学習後の EMA) を Ta=8 (既定) と Ta=2 で。
  4 スロット x 30 コア (D3IL_CPU_OFFSET)。空きメモリの多い GPU に載せる。
  energy_inf の pilot が走っている間は評価を始めない (コアが重なるため)。
出力: logs/em_all/<task>/<mod>_s<seed>/ (学習), 同ディレクトリの eval_ta{8,2}.log
"""
import os, sys, time, subprocess

ROOT = "/mnt/data1/home/rkawata/work/EnergyMatching/experiments/d3il"
PY = sys.executable
OUT = "logs/em_all"
LOG = os.path.join(ROOT, OUT, "orchestrate.log")

# task: (config, eval overrides)
STATE = {
    "sorting_4": ("sorting_4_config", ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=18"]),
    "sorting_6": ("sorting_6_config", ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=20"]),
    "stacking": ("stacking_config", ["simulation.render=False", "simulation.n_contexts=60",
                                     "simulation.n_trajectories_per_context=18"]),
    "inserting": ("inserting_config", ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=6"]),
}
VISION = {
    "sorting_4": ("sorting_4_vision_config", ["simulation.max_steps_per_episode=700"]),
    "sorting_6": ("sorting_6_vision_config", ["simulation.max_steps_per_episode=1200"]),
    "stacking": ("stacking_vision_config", ["simulation.max_steps_per_episode=1200"]),
}
VIS_EVAL = ["simulation.render=False", "simulation.if_vision=True", "simulation.n_contexts=60",
            "simulation.n_trajectories_per_context=6"]
STATE_SEEDS = [0, 1, 2]
VISION_SEEDS = [0]
TAS = [8, 2]
GPUS = [0, 1, 2, 3, 4, 5]
MAX_TRAIN_PER_GPU = 2
EVAL_SLOTS = [0, 30, 60, 90]


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def pilot_running():
    return subprocess.run(["pgrep", "-f", "orchestrate.py pilot"], capture_output=True).returncode == 0


def train_jobs():
    jobs = []
    for task, (cfg, _) in VISION.items():          # 画像は長いので先に
        for s in VISION_SEEDS:
            jobs.append(dict(task=task, mod="vision", seed=s, cfg=cfg, agent="em_vision_agent", name="em_vision"))
    for task, (cfg, _) in STATE.items():
        for s in STATE_SEEDS:
            jobs.append(dict(task=task, mod="state", seed=s, cfg=cfg, agent="em_agent", name="em"))
    for j in jobs:
        j["dir"] = os.path.join(OUT, j["task"], f"{j['mod']}_s{j['seed']}")
    return jobs


def eval_jobs(j):
    ev = []
    for ta in TAS:
        if j["mod"] == "state":
            ov, n_cores, need = STATE[j["task"]][1], 30, 25000
        else:
            ov, n_cores, need = VISION[j["task"]][1] + VIS_EVAL, 20, 36000
        ev.append(dict(job=j, ta=ta, ov=ov + [f"agents.exec_horizon={ta}"], n_cores=n_cores, need=need,
                       log=os.path.join(ROOT, j["dir"], f"eval_ta{ta}.log")))
    return ev


def eval_done(e):
    return os.path.exists(e["log"]) and "RESULT" in open(e["log"]).read()


def main():
    trains = [j for j in train_jobs()]
    pending_train = [j for j in trains if not os.path.exists(os.path.join(ROOT, j["dir"], "last_em.pth"))]
    pending_eval = [e for j in trains if j not in pending_train for e in eval_jobs(j) if not eval_done(e)]
    running_train, running_eval = {}, {}       # proc -> job ; slot -> (proc, eval)
    failed = []
    log(f"start: {len(pending_train)} trainings, {len(pending_eval)} evals already eligible")
    while pending_train or running_train or pending_eval or running_eval:
        # --- 終了チェック
        for p in list(running_train):
            if p.poll() is not None:
                j = running_train.pop(p)
                ok = p.returncode == 0 and os.path.exists(os.path.join(ROOT, j["dir"], "last_em.pth"))
                log(f"train done {j['dir']} rc={p.returncode} ok={ok}")
                if ok:
                    pending_eval += eval_jobs(j)
                else:
                    failed.append(j["dir"])
        for slot in list(running_eval):
            p, e = running_eval[slot]
            if p.poll() is not None:
                del running_eval[slot]
                ok = p.returncode == 0 and eval_done(e)
                log(f"eval done {e['job']['dir']} ta{e['ta']} rc={p.returncode} ok={ok}")
                if not ok:
                    failed.append(f"{e['job']['dir']}:ta{e['ta']}")
        # --- 学習の投入
        load = {g: 0 for g in GPUS}
        for j in running_train.values():
            load[j["gpu"]] += 1
        while pending_train:
            g = min(GPUS, key=lambda g: (load[g], g))
            if load[g] >= MAX_TRAIN_PER_GPU:
                break
            j = pending_train.pop(0)
            j["gpu"] = g
            load[g] += 1
            os.makedirs(os.path.join(ROOT, j["dir"]), exist_ok=True)
            cmd = [PY, "run.py", f"--config-name={j['cfg']}", f"agents={j['agent']}", f"agent_name={j['name']}",
                   f"seed={j['seed']}", "wandb.entity=none", "wandb.project=d3il", "+skip_sim=true",
                   f"hydra.run.dir={j['dir']}"]
            lf = open(os.path.join(ROOT, j["dir"] + ".train.log"), "w")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_GL="egl"))
            running_train[p] = j
            log(f"train start {j['dir']} gpu {g}")
        # --- 評価の投入 (pilot 終了後)
        if pending_eval and not pilot_running():
            for slot in EVAL_SLOTS:
                if slot in running_eval or not pending_eval:
                    continue
                free = gpu_free()
                e = pending_eval[0]
                g = max(GPUS, key=lambda g: free[g])
                if free[g] < e["need"]:
                    break
                pending_eval.pop(0)
                cmd = [PY, "analysis/eval_vision_ckpt.py", e["job"]["dir"], "last_em.pth", str(e["n_cores"]), *e["ov"]]
                lf = open(e["log"], "w")
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_GL="egl",
                                              D3IL_CPU_OFFSET=str(slot), OMP_NUM_THREADS="1"))
                running_eval[slot] = (p, e)
                log(f"eval start {e['job']['dir']} ta{e['ta']} gpu {g} cores {slot}-{slot + 29}")
                time.sleep(60)          # GPU メモリが埋まるのを待ってから次を判断
        time.sleep(30)
    log(f"ALLDONE failed={failed}")


if __name__ == "__main__":
    main()
