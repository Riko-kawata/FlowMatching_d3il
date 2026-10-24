"""Flow 学習の重みで D3IL タスクを推論し、カメラ画像を mp4 に書き出す。

experiments/d3il のコード・データをそのまま使う (d3il 側のファイルは変更しない)。
重みは dl_kouza/runs/<task>/<mod>_s<seed>/em_flow_step*.pth、学習時の設定は d3il/logs/em_all の同じ run の
.hydra/config.yaml (d3il のコードで読める形) から読む。
評価ループは各タスクの Sim.test_agent (n_cores=1) をそのまま呼び、環境の reset / step を
このプロセス内だけで包んで、毎 step 後に俯瞰カメラ (bp_cam) と手先カメラ (inhand_cam) を描画する。
描画は MjData のコピーに対して行うので、物理 (= 推論結果) には影響しない。

Usage (d3il_env で):
  python dl_kouza/scripts/record_flow_videos.py --task sorting_4 --mod vision --seed 0 \
      --contexts 0 1 2 --ta 2 --out dl_kouza/videos
出力: <out>/<task>_<mod>_s<seed>/ctx<c>_k<k>_<success|fail>.mp4 と results.json
"""
import os
import sys
import json
import argparse
import subprocess

import numpy as np
import cv2

D3IL = "/mnt/data1/home/rkawata/work/EnergyMatching/experiments/d3il"
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
FLOW_CKPT = {"state": "em_flow_step100000.pth", "vision": "em_flow_step60000.pth"}
# orchestrate_flow.py の評価条件と同じ上書き
VISION_EVAL = {
    "sorting_4": ["simulation.max_steps_per_episode=700"],
    "sorting_6": ["simulation.max_steps_per_episode=1200"],
    "stacking": ["simulation.max_steps_per_episode=1200"],
}


class Recorder:
    """環境クラスの reset / step を包み、1 エピソード = 1 本の mp4 を書く。"""

    def __init__(self, out_dir, size, fps, title):
        self.out_dir, self.size, self.fps, self.title = out_dir, size, fps, title
        self.ep = 0
        self.proc = None
        self.results = []
        self.ctx_list = None
        self.n_traj = 1

    def _open(self, env):
        import mujoco
        if getattr(env, "_rec_renderer", None) is None:
            env._rec_renderer = mujoco.Renderer(env.scene.model, self.size, self.size)
            env._rec_data = mujoco.MjData(env.scene.model)
        ctx = self.ctx_list[self.ep // self.n_traj]
        k = self.ep % self.n_traj
        self.cur = dict(ctx=int(ctx), k=int(k), steps=0, success=False)
        self.tmp = os.path.join(self.out_dir, f".tmp_ep{self.ep}.mp4")
        w, h = 2 * self.size, self.size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-movflags", "+faststart", self.tmp], stdin=subprocess.PIPE)

    def _frame(self, env, info=None):
        import mujoco
        model, data, rdata = env.scene.model, env.scene.data, env._rec_data
        rdata.qpos[:], rdata.qvel[:] = data.qpos, data.qvel
        rdata.mocap_pos[:], rdata.mocap_quat[:] = data.mocap_pos, data.mocap_quat
        mujoco.mj_forward(model, rdata)
        imgs = []
        for cam in (env.bp_cam, env.inhand_cam):
            env._rec_renderer.update_scene(rdata, camera=cam.name)
            imgs.append(cv2.cvtColor(env._rec_renderer.render(), cv2.COLOR_RGB2BGR))
        img = np.concatenate(imgs, axis=1)
        lines = [self.title, f"ctx {self.cur['ctx']}  step {self.cur['steps']}"]
        if info is not None and info.get("success"):
            lines.append("SUCCESS")
        for j, txt in enumerate(lines):
            org = (8, 22 + 22 * j)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for j, txt in enumerate(["bp_cam (overhead)", "inhand_cam"]):
            org = (j * self.size + 8, self.size - 10)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self.proc.stdin.write(img.tobytes())

    def close(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        self.proc.wait()
        tag = "success" if self.cur["success"] else "fail"
        path = os.path.join(self.out_dir, f"ctx{self.cur['ctx']:02d}_k{self.cur['k']}_{tag}.mp4")
        os.replace(self.tmp, path)
        self.cur["video"] = os.path.basename(path)
        self.results.append(self.cur)
        print(f"wrote {path} steps={self.cur['steps']}", flush=True)
        self.proc = None
        self.ep += 1

    def patch(self, cls):
        rec = self
        orig_reset, orig_step = cls.reset, cls.step

        def reset(env, *a, **kw):
            rec.close()
            obs = orig_reset(env, *a, **kw)
            rec._open(env)
            rec._frame(env)
            return obs

        def step(env, *a, **kw):
            out = orig_step(env, *a, **kw)
            info = out[3]
            rec.cur["steps"] += 1
            rec.cur["success"] = bool(info.get("success", False))
            rec._frame(env, info)
            return out

        cls.reset, cls.step = reset, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sorting_4", "sorting_6", "stacking", "inserting"])
    ap.add_argument("--mod", required=True, choices=["state", "vision"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_traj", type=int, default=1, help="各 context の試行回数")
    ap.add_argument("--ta", type=int, default=2, help="exec_horizon (評価時の Ta)")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.chdir(D3IL)
    sys.path.insert(0, D3IL)
    import torch
    import hydra
    import wandb
    from omegaconf import OmegaConf
    import importlib

    OmegaConf.register_new_resolver("add", lambda *n: sum(n))
    run_dir = os.path.join("logs/em_all", args.task, f"{args.mod}_s{args.seed}")
    ckpt = FLOW_CKPT[args.mod]
    weights_dir = os.path.join(RUNS, args.task, f"{args.mod}_s{args.seed}")
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    ov = [f"agents.exec_horizon={args.ta}", "simulation.render=False", "simulation.n_cores=1",
          f"simulation.n_contexts={max(args.contexts) + 1}", f"simulation.n_trajectories_per_context={args.n_traj}"]
    if args.mod == "vision":
        ov += VISION_EVAL[args.task] + ["simulation.if_vision=True"]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(ov))

    wandb.init(mode="disabled")
    if args.mod == "vision":
        # em_all の画像 run (10/01-10/04) の後に dataset が LazyImageStore 版に置き換わり、
        # EMVisionAgent._compact が参照する bp_cam_imgs が無くなった。_compact は学習用の画像詰め直しで、
        # 推論に要る正規化統計 (EMAgent.__init__ が actions から計算) には関係しないので、ここでは飛ばす。
        from agents.em_vision_agent import EMVisionAgent
        EMVisionAgent._compact = lambda self, ds: None
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(weights_dir, sv_name=ckpt)
    sim = hydra.utils.instantiate(cfg.simulation)

    # 指定 context だけ回す: test_agent は 0..n_contexts-1 を回すので、test_contexts を並べ替えて先頭に置く
    sim.test_contexts = [sim.test_contexts[c] for c in args.contexts]
    sim.n_contexts = len(args.contexts)

    sim_mod = importlib.import_module(type(sim).__module__)
    sim_mod.assign_process_to_cpu = lambda pid, cpus: None   # n_cores=1 だとコア 0 に固定されるのを避ける
    env_cls = {"sorting_4": "Sorting_Env", "sorting_6": "Sorting_Env", "stacking": "CubeStacking_Env",
               "inserting": "Gate_Insertion_Env"}[args.task]

    name = f"{args.task}_{args.mod}_s{args.seed}"
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    rec = Recorder(out_dir, args.size, args.fps, f"{args.task} / {args.mod} / seed {args.seed} / Flow / Ta={args.ta}")
    rec.ctx_list, rec.n_traj = args.contexts, args.n_traj
    rec.patch(getattr(sim_mod, env_cls))

    sim.test_agent(agent)
    rec.close()

    summary = dict(task=args.task, mod=args.mod, seed=args.seed, run_dir=os.path.join(D3IL, run_dir), ckpt=os.path.join(weights_dir, ckpt),
                   exec_horizon=args.ta, fps=args.fps, overrides=ov, episodes=rec.results,
                   success_rate=float(np.mean([r["success"] for r in rec.results])))
    json.dump(summary, open(os.path.join(out_dir, "results.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
