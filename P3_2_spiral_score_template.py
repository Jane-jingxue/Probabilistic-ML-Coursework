#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 2/3: Score-based modeling on a 2D spiral (P3.2)

This skeleton covers:
- P3.2.1 forward VP-SDE (Euler–Maruyama)
- P3.2.2 reverse process using the TRUE score (mixture score)  [TODO]
- P3.2.3 learn score with Hyvärinen objective + Hutchinson divergence  [TODO divergence helper]
- P3.2.4 generation with learned score

Students must fill:
- true mixture score (responsibilities)
- Hutchinson divergence helper
- score net design tweaks and reporting/plots

Run:
  python P3_2_spiral_score_template.py
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class DiffusionConfig:
    #  Spiral mixture parameters  
    M: int = 200
    a: float = 0.2
    b: float = 0.15
    theta_min: float = 0.0
    theta_max: float = 4.0 * math.pi
    sigma0: float = 0.06  # Initial data noise

    beta0: float = 0.001
    beta1: float = 12
    K: int = 500  # Discretization steps

    # Score learning
    steps: int = 10000     # Training steps
    batch_size: int = 256
    lr: float = 1e-3

    outdir: str = "outputs_q2_spiral"
    seed: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spiral_means(cfg: DiffusionConfig) -> np.ndarray:
    """Generate the means mu_m for the spiral mixture."""
    thetas = np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)
    r = cfg.a + cfg.b * thetas
    mu = np.stack([r * np.cos(thetas), r * np.sin(thetas)], axis=1).astype(np.float32)
    return mu


def sample_p0(cfg: DiffusionConfig, n: int) -> np.ndarray:
    """Sample x0 from the spiral mixture p0(x)."""
    mu = spiral_means(cfg)
    idx = np.random.randint(0, cfg.M, size=n)
    eps = np.random.randn(n, 2).astype(np.float32) * cfg.sigma0
    return mu[idx] + eps


def beta(t: float, beta0, beta1) -> float:
    """Linear noise schedule beta(t)."""
    return beta0 + (beta1 - beta0) * t


def alpha_bar(K, beta0, beta1) -> np.ndarray:
    dt = 1.0 / K
    alphas = []
    # k=1 to K corresponds to steps t_0 to t_{K-1}
    for k in range(1, K + 1):
        t_prev = (k - 1) / K
        b = beta(t_prev, beta0, beta1)
        a = math.exp(-b * dt)
        alphas.append(a)
    
    alphas = np.array(alphas)
    alphas_cum = np.cumprod(alphas)
    # Prepend 1.0 for k=0
    return np.concatenate(([1.0], alphas_cum))


def forward_euler(
    cfg: DiffusionConfig, x0: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
     Forward Euler-Maruyama for VP-SDE. 
    x_{k+1} = x_k - 0.5*beta_k*x_k*dt + sqrt(beta_k*dt)*xi
    """
    dt = 1.0 / cfg.K
    traj = [x0]
    x = x0.copy()
    
    for k in range(cfg.K):
        # Time t_k = k/K
        t = k / cfg.K
        b = beta(t, cfg.beta0, cfg.beta1)
        
        # Euler-Maruyama update
        drift = -0.5 * b * x * dt
        diffusion = math.sqrt(b * dt) * np.random.randn(*x.shape)
        
        x = x + drift + diffusion
        traj.append(x)
        
    return traj, x


def pk_mixture_params(cfg: DiffusionConfig, k: int) -> Tuple[np.ndarray, float]:
    """
     Compute parameters (mu_k, var_k) for the marginal p_k(x).
    p_k(x) is a mixture of Gaussians with means shrunk by sqrt(alpha_bar).
    """
    ab_full = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)
    ab_k = ab_full[k]
    
    mu_0 = spiral_means(cfg)

    mu_k = math.sqrt(ab_k) * mu_0
    var_k = ab_k * (cfg.sigma0**2) + (1 - ab_k)
    
    return mu_k, var_k


def true_score_mixture(x: np.ndarray, means: np.ndarray, var: float) -> np.ndarray:
    """
     Compute the true score \nabla_x log p_k(x) for a GMM.
    p(x) = sum_m pi_m N(x; mu_m, var*I)
    score(x) = sum_m w_m(x) * (mu_m - x) / var
    where w_m(x) are the softmax responsibilities.
    """
    # x: (B, 2), means: (M, 2)
    x_exp = x[:, None, :]      # (B, 1, 2)
    mu_exp = means[None, :, :] # (1, M, 2)
    
    # Squared Euclidean distance ||x - mu_m||^2
    dists_sq = np.sum((x_exp - mu_exp)**2, axis=2) # (B, M)
    
    # Log-likelihoods of each component (ignoring constants)
    log_probs = -0.5 * dists_sq / var
    
    # Softmax to get responsibilities pi(m|x)
    # Use max subtraction for numerical stability
    max_log = np.max(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs - max_log)
    denom = np.sum(probs, axis=1, keepdims=True) + 1e-10
    resps = probs / denom  # (B, M)
    
    # Score is weighted sum of component scores: -(x - mu)/var = (mu - x)/var
    # (B, M, 1) * (B, M, 2) -> sum over M -> (B, 2)
    diffs = (mu_exp - x_exp) / var
    score = np.sum(resps[:, :, None] * diffs, axis=1)
    
    return score


def reverse_euler(cfg: DiffusionConfig, xK: np.ndarray, score_fn) -> List[np.ndarray]:
    """
     Reverse Euler-Maruyama.
    Start at x_K ~ N(0,I) (or learned dist) and go backward to x_0.
    In discrete reverse time (dt > 0):
    x_{k-1} = x_k + [0.5*beta*x + beta*score] * dt + sqrt(beta*dt) * z
    """
    dt = 1.0 / cfg.K
    traj = [xK]
    x = xK.copy()
    
    # Iterate backwards: K-1, ..., 0
    for k in range(cfg.K - 1, -1, -1):
        # Time t corresponds to step k+1 (current state x is at t_{k+1})
        t_curr = (k + 1) / cfg.K
        b = beta(t_curr, cfg.beta0, cfg.beta1)
        
        # Get score at current state x and time k+1
        s = score_fn(x, k + 1)
        
        # Reverse drift: (0.5*beta*x + beta*score)
        drift = (0.5 * b * x + b * s) * dt
        
        # Diffusion: sqrt(beta)*dw
        noise = np.random.randn(*x.shape) * math.sqrt(dt)
        diffusion = math.sqrt(b) * noise
        
        x = x + drift + diffusion
        traj.append(x)
        
    return traj


class ScoreNet(nn.Module):
    """Simple MLP to approximate the score s(x, t)."""
    def __init__(self, hidden: int = 128):
        super().__init__()
        # Input: 2 (x) + 1 (t) = 3 dimensions
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(), 
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2) # Output: 2D vector
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Concatenate x and t
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def divergence_hutchinson(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
     Exact divergence for 2D vector field using autograd.
    Computes div(s) = ds1/dx1 + ds2/dx2.
    """
    s1 = s[:, 0]
    s2 = s[:, 1]
    
    # create_graph=True allows higher-order derivatives for backprop
    # grad_outputs=ones because we want sum of gradients
    g1 = torch.autograd.grad(s1.sum(), x, create_graph=True)[0]
    g2 = torch.autograd.grad(s2.sum(), x, create_graph=True)[0]
    
    return g1[:, 0] + g2[:, 1]


def train_score(cfg: DiffusionConfig, model: nn.Module, device: str) -> None:
    """
     Train using Hyvärinen score matching objective.
    J = E [ 0.5 ||s||^2 + div(s) ]
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    bar = tqdm(range(1, cfg.steps + 1))
    loss_history = []

    for step in range(1, cfg.steps + 1):
        # 1. Sample time t uniformly
        k = np.random.randint(1, cfg.K + 1)
        tval = k / cfg.K
        
        # 2. Sample data from p_k(x) mixture
        means_k, var_k = pk_mixture_params(cfg, k)
        
        # Sample mixture component indices
        m_idx = np.random.randint(0, cfg.M, size=cfg.batch_size)
        mu = means_k[m_idx]
        
        # Add noise to get x_k
        xk = mu + np.random.randn(cfg.batch_size, 2).astype(np.float32) * math.sqrt(var_k)

        # 3. Prepare tensors
        x = torch.tensor(xk, device=device, requires_grad=True).float()
        t = torch.full((cfg.batch_size, 1), float(tval), device=device).float()

        opt.zero_grad()
        
        # 4. Compute Score and Divergence
        s = model(x, t)
        div = divergence_hutchinson(s, x)
        
        # 5. Hyvärinen Objective
        norm_sq = 0.5 * torch.sum(s**2, dim=1)
        loss = (norm_sq + div).mean()
        
        loss.backward()
        opt.step()
        
        loss_history.append(loss.item())
        if step % 100 == 0:
            bar.set_description(f"[train] loss={loss.item():.4f}")
        bar.update(1)

    bar.close()
    
    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Score Matching Loss")
    plt.savefig(os.path.join(cfg.outdir, "training_loss.png"))
    plt.close()


def score_true(x: np.ndarray, k: int) -> np.ndarray:
    # This wrapper is actually hard to implement cleanly without passing `cfg`
    # because pk_mixture_params requires `cfg`.
    # To fix this, we will use a global `CFG_GLOBAL` or simply pass the cfg via closure in main.
    # The implementation below is a placeholder, logic is handled via closures in main.
    pass


def scatter(x: np.ndarray, title: str, path: str):
    """Helper to plot 2D scatter points."""
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5)
    plt.title(title)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    cfg = DiffusionConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)


    print("Running Forward Process...")
    x0 = sample_p0(cfg, n=2000)
    
    # Save original data plot
    scatter(x0, "Data x0", os.path.join(cfg.outdir, "Spiral_distribution.png"))

    traj_fwd, _ = forward_euler(cfg, x0)
    
    # Plot snapshots
    plot_steps = [0, 20, 50, 100, cfg.K]
    for k in plot_steps:
        if k < len(traj_fwd):
            scatter(traj_fwd[k], f"Forward Process k={k}", 
                    os.path.join(cfg.outdir, f"forward_k{k}.png"))

    #P3.2.2 Reverse Process (True Score)
    print("Running Reverse Process (True Score)...")
    
    # Wrapper to adapt signature for reverse_euler
    def true_score_wrapper(x_np, k):
        # Determine mixture params at step k
        mu_k, var_k = pk_mixture_params(cfg, k)
        return true_score_mixture(x_np, mu_k, var_k)

    # Start from noise (standard normal)

    xK = np.random.randn(2000, 2).astype(np.float32)
    traj_rev_true = reverse_euler(cfg, xK, true_score_wrapper)
    
    scatter(traj_rev_true[-1], "Reverse Process (True Score) Final", 
            os.path.join(cfg.outdir, "reverse_true_final.png"))


    print("Training Score Network...")
    model = ScoreNet(hidden=128).to(device)
    train_score(cfg, model, device)


    print("Running Reverse Process (Learned Score)...")
    
    def learned_score_wrapper(x_np, k):
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_np, device=device).float()
            # t = k/K
            t_t = torch.full((x_np.shape[0], 1), float(k / cfg.K), device=device).float()
            return model(x_t, t_t).cpu().numpy()

    xK_gen = np.random.randn(2000, 2).astype(np.float32)
    traj_rev_learn = reverse_euler(cfg, xK_gen, learned_score_wrapper)
    
    scatter(traj_rev_learn[-1], "Reverse Process (Learned Score) Final", 
            os.path.join(cfg.outdir, "reverse_learned_final.png"))


    print("Visualizing Vector Field...")
    grid_x = np.linspace(-2.5, 2.5, 20)
    grid_y = np.linspace(-2.5, 2.5, 20)
    xx, yy = np.meshgrid(grid_x, grid_y)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    
    model.eval()
    with torch.no_grad():
        t_val = 0.1
        xy_t = torch.tensor(xy, device=device)
        tt = torch.full((xy.shape[0], 1), t_val, device=device)
        uv = model(xy_t, tt).cpu().numpy()
        
    plt.figure(figsize=(6, 6))
    plt.quiver(xx, yy, uv[:,0].reshape(20,20), uv[:,1].reshape(20,20), scale=50)
    plt.title(f"Learned Score Field (t={t_val})")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.savefig(os.path.join(cfg.outdir, "score_field_t01.png"))
    plt.close()

    # Save model weights
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "score_net_spiral.pt"))
    print("Saved weights to:", os.path.join(cfg.outdir, "score_net_spiral.pt"))


if __name__ == "__main__":
    main()