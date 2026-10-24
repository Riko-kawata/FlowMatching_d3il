"""Evaluate a saved checkpoint (e.g. last_*.pth) with the run's own simulation config (any task, state or vision).

Usage: python analysis/eval_vision_ckpt.py <run_dir> <ckpt_name> [n_cores] [key=value ...]
  e.g. ... runs/x em_flow_step40000.pth 15 agents.exec_horizon=4
Re-instantiates the agent from <run_dir>/.hydra/config.yaml (dataset is reloaded for the scaler).
"""
import sys, os, logging
sys.path.insert(0, os.getcwd())  # script lives in analysis/, agents/ is resolved from cwd
import hydra, wandb, torch
from omegaconf import OmegaConf


def main():
    OmegaConf.register_new_resolver("add", lambda *n: sum(n))
    logging.basicConfig(level=logging.INFO)

    run_dir, ckpt = sys.argv[1], sys.argv[2]
    n_cores = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    cfg.simulation.n_cores = n_cores
    overrides = sys.argv[4:]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    wandb.init(mode="disabled")
    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(run_dir, sv_name=ckpt)
    sim = hydra.utils.instantiate(cfg.simulation)
    sr, _ = sim.test_agent(agent)
    # EVAL_NPZ=<path> でエピソードごとの結果 (sim.last_results) を保存する (ペア比較用)
    npz = os.environ.get("EVAL_NPZ")
    if npz and getattr(sim, "last_results", None) is not None:
        import numpy as np
        np.savez(npz, **sim.last_results)
        print(f"saved per-episode results -> {npz}")
    print(f"RESULT run={run_dir} ckpt={ckpt} overrides={overrides} ret={float(sr.float().mean()) if torch.is_tensor(sr) else float(sr):.4f} (成功率はログの Successrate 行を見ること)")


if __name__ == "__main__":  # sim uses mp 'spawn'
    main()
