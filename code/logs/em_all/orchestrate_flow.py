"""Flow 学習の重みの成功率を全タスクで測る (EM_ALL_TASKS_RESULTS.md の全タスク)。
Usage (experiments/d3il から): nohup python logs/em_all/orchestrate_flow.py > logs/em_all/orchestrate_flow.out 2>&1 &

1. 画像入力の学習 seed 1, 2 を追加 (Sorting4 / Sorting6 / Stacking, Flow 60k step)。
2. 全 21 run (状態 12 + 画像 9) x Ta {8, 2} を同じ評価条件で評価し、
   エピソードごとの結果を npz に残す (EVAL_NPZ)。評価の乱数はワーカー番号で決まる。
- 他ユーザーが全 GPU を使っているので控えめに: 学習は GPU 0-2 に 1 本ずつ、評価は 3 並列 (各 30 コア)、
  空きメモリが足りる GPU にだけ載せる。
出力: <run>/flow_ta{8,2}.{log,npz}
"""
import os, sys, time, subprocess

ROOT = "/mnt/data1/home/rkawata/work/EnergyMatching/experiments/d3il"
PY = sys.executable
OUT = "logs/em_all"
LOG = os.path.join(ROOT, OUT, "orchestrate_flow.log")

STATE = {
    "sorting_4": ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=18"],
    "sorting_6": ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=20"],
    "stacking": ["simulation.render=False", "simulation.n_contexts=60", "simulation.n_trajectories_per_context=18"],
    "inserting": ["simulation.n_contexts=60", "simulation.n_trajectories_per_context=6"],
}
VISION = {
    "sorting_4": ("sorting_4_vision_config", ["simulation.max_steps_per_episode=700"]),
    "sorting_6": ("sorting_6_vision_config", ["simulation.max_steps_per_episode=1200"]),
    "stacking": ("stacking_vision_config", ["simulation.max_steps_per_episode=1200"]),
}
VIS_EVAL = ["simulation.render=False", "simulation.if_vision=True", "simulation.n_contexts=60",
            "simulation.n_trajectories_per_context=6"]
STATE_SEEDS = [0, 1, 2]
VISION_SEEDS = [0, 1, 2]
NEW_VISION_SEEDS = [1, 2]
TAS = [8, 2]
TRAIN_GPUS = [0, 1, 2]
EVAL_GPUS = [0, 1, 2, 3, 4, 5]
EVAL_SLOTS = [0, 30, 60]
FLOW_CKPT = {"state": "em_flow_step100000.pth", "vision": "em_flow_step60000.pth"}


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def gpu_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def run_dir(task, mod, seed):
    return os.path.join(OUT, task, f"{mod}_s{seed}")


def trained(d):
    return os.path.exists(os.path.join(ROOT, d, "last_em.pth"))


def eval_jobs(task, mod, seed):
    d = run_dir(task, mod, seed)
    jobs = []
    for ta in TAS:
        if mod == "state":
            ov, n_cores, need = STATE[task], 30, 24000
        else:
            ov, n_cores, need = VISION[task][1] + VIS_EVAL, 12, 24000
        base = os.path.join(ROOT, d, f"flow_ta{ta}")
        jobs.append(dict(dir=d, ta=ta, ckpt=FLOW_CKPT[mod], ov=ov + [f"agents.exec_horizon={ta}"],
                         n_cores=n_cores, need=need, log=base + ".log", npz=base + ".npz"))
    return jobs


def eval_done(e):
    return os.path.exists(e["npz"])


def main():
    trains = [dict(task=t, seed=s, cfg=VISION[t][0], dir=run_dir(t, "vision", s))
              for s in NEW_VISION_SEEDS for t in VISION]
    pending_train = [j for j in trains if not trained(j["dir"])]
    # 評価対象: 学習済みのもの。未学習の画像 run は学習終了時に追加する
    all_runs = [(t, "state", s) for t in STATE for s in STATE_SEEDS] + \
               [(t, "vision", s) for s in VISION_SEEDS for t in VISION]
    pending_eval = [e for r in all_runs if trained(run_dir(*r)) for e in eval_jobs(*r) if not eval_done(e)]
    running_train, running_eval, failed = {}, {}, []
    log(f"start: {len(pending_train)} trainings, {len(pending_eval)} evals")
    while pending_train or running_train or pending_eval or running_eval:
        for p in list(running_train):
            if p.poll() is not None:
                j = running_train.pop(p)
                ok = p.returncode == 0 and trained(j["dir"])
                log(f"train done {j['dir']} rc={p.returncode} ok={ok}")
                if ok:
                    pending_eval += [e for e in eval_jobs(j["task"], "vision", j["seed"]) if not eval_done(e)]
                else:
                    failed.append(j["dir"])
        for slot in list(running_eval):
            p, e = running_eval[slot]
            if p.poll() is not None:
                del running_eval[slot]
                ok = p.returncode == 0 and eval_done(e)
                log(f"eval done {e['dir']} ta{e['ta']} rc={p.returncode} ok={ok}")
                if not ok:
                    failed.append(f"{e['dir']}:ta{e['ta']}")
        # 学習: GPU 0-2 に 1 本ずつ
        busy = {j["gpu"] for j in running_train.values()}
        for g in TRAIN_GPUS:
            if g in busy or not pending_train:
                continue
            j = pending_train.pop(0)
            j["gpu"] = g
            os.makedirs(os.path.join(ROOT, j["dir"]), exist_ok=True)
            cmd = [PY, "run.py", f"--config-name={j['cfg']}", "agents=em_vision_agent", "agent_name=em_vision",
                   f"seed={j['seed']}", "wandb.entity=none", "wandb.project=d3il", "+skip_sim=true",
                   f"hydra.run.dir={j['dir']}"]
            lf = open(os.path.join(ROOT, j["dir"] + ".train.log"), "w")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_GL="egl"))
            running_train[p] = j
            log(f"train start {j['dir']} gpu {g}")
        # 評価: 空きスロットに、空きメモリが足りる GPU があれば 1 本ずつ
        for slot in EVAL_SLOTS:
            if slot in running_eval or not pending_eval:
                continue
            free = gpu_free()
            used = {running_eval[s][1]["gpu"] for s in running_eval}
            cand = [g for g in EVAL_GPUS if g not in used and free[g] >= pending_eval[0]["need"]]
            if not cand:
                break
            g = max(cand, key=lambda g: free[g])
            e = pending_eval.pop(0)
            e["gpu"] = g
            cmd = [PY, "analysis/eval_vision_ckpt.py", e["dir"], e["ckpt"], str(e["n_cores"]), *e["ov"]]
            lf = open(e["log"], "w")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), MUJOCO_GL="egl",
                                          D3IL_CPU_OFFSET=str(slot), OMP_NUM_THREADS="1", EVAL_NPZ=e["npz"]))
            running_eval[slot] = (p, e)
            log(f"eval start {e['dir']} ta{e['ta']} gpu {g} cores {slot}-{slot + 29}")
            time.sleep(90)          # GPU メモリが埋まるのを待ってから次を判断
        time.sleep(30)
    log(f"ALLDONE failed={failed}")


if __name__ == "__main__":
    main()
