"""Inference-only sampler experiments; the original EM sampler stays untouched."""
import math

import torch


@torch.no_grad()
def integrate_ablation(agent, c, solver="euler", ode_time=None, sde_time=None):
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unknown solver: {solver}")
    split = ode_time is not None or sde_time is not None
    if split and (ode_time is None or sde_time is None):
        raise ValueError("Specify both ode_time and sde_time")
    dt = 1.0 / agent.num_steps
    total = agent.sample_T if not split else ode_time + sde_time
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Sampling duration must be finite and positive")
    if split and (ode_time < 0 or sde_time <= 0 or agent.sample_eps_max <= 0):
        raise ValueError("Split sampling requires ODE >= 0, SDE > 0 and positive temperature")
    n = round(total / dt)
    if not math.isclose(n * dt, total, abs_tol=1e-9):
        raise ValueError("Duration must be a multiple of dt; do not silently round experiments")
    if split and not math.isclose(round(ode_time / dt) * dt, ode_time, abs_tol=1e-9):
        raise ValueError("ODE boundary must be a multiple of dt")
    x = torch.randn(c.shape[0], agent.H, agent.model.action_dim, device=c.device)
    for i in range(n):
        t = i * dt
        # Translate the existing temperature ramp without changing its width or height.
        # With the defaults it rises over 0.2 time units, then remains at eps_max.
        schedule_t = t if not split else t - ode_time + agent.tau_star
        eps = agent._eps_at(schedule_t, agent.sample_eps_max)
        v = agent.model.velocity(x, c)
        proposal = x + v * dt
        noise = None
        if eps > 0:
            noise = (2 * eps * dt) ** 0.5 * torch.randn_like(x)
            proposal = proposal + noise
        if solver == "heun":
            # One Wiener increment per step, reused in the predictor and corrector.
            v_next = agent.model.velocity(proposal, c)
            x = x + 0.5 * dt * (v + v_next)
            if noise is not None:
                x = x + noise
        else:
            x = proposal
    return x
