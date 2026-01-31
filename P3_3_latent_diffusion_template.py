#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 3/3: VAE + latent diffusion for MNIST (P3.3)
Based on the exam instructions. fileciteturn14file0

This skeleton covers:
- P3.3.2 encode MNIST -> latent dataset {z_i} (using z = mu(x))
- P3.3.3 learn latent score with Hyvärinen + Hutchinson divergence
- P3.3.4 reverse sampling in latent space + decode to images
- P3.3.5 compare VAE-only vs latent diffusion

Students must fill:
- Hutchinson divergence helper
- (optionally) better sampling / plots / hyperparams
- (optional Q4) conditional extension s(z,t,y)

Run:
  python P3_P3.3_latent_diffusion_template.py
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


try:
    from P3_1_vae_template import VAE
    from P3_2_spiral_score_template import (
        ScoreNet,
        divergence_hutchinson,
        reverse_euler,
        alpha_bar,
    )
except ImportError:
    print("Error: P3_1_vae_template.py or P3_2_spiral_score_template.py not found.")
    exit(1)


@dataclass
class LatentConfig:
    vae_weights: str = "outputs_q1_vae/vae_mnist.pt"
    latent_dim: int = 2          
    hidden_dim: int = 400        
    
    score_hidden: int = 256      # Size of score network

    # Diffusion parameters
    beta0: float = 0.001
    beta1: float = 9
    K: int = 400

    # Training parameters
    steps: int = 10000
    batch_size: int = 512
    lr: float = 1e-3

    outdir: str = "outputs_P3.3_latent"
    seed: int = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise(z0, k, ab, cfg: LatentConfig) -> np.ndarray:
    """
    Sample z_k ~ q(z_k | z_0)
    z_k = sqrt(alpha_bar_k) * z_0 + sqrt(1 - alpha_bar_k) * eps
    """
    ab_k = ab[k] # alpha_bar at index k
    eps = np.random.randn(*z0.shape).astype(np.float32)
    zk = math.sqrt(ab_k) * z0 + math.sqrt(1 - ab_k) * eps
    return zk


def train_score(Zs, cfg: LatentConfig, model: nn.Module, device: str) -> None:
    #  Train score model with Hyvärinen objective 
    ab = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    
    losses = []

    bar = tqdm(range(1, cfg.steps + 1))
    for step in range(1, cfg.steps + 1):
        # 1. Sample batch from latent dataset
        idx = np.random.randint(0, Zs.shape[0], size=cfg.batch_size)
        z0 = Zs[idx]

        # 2. Sample random time step
        k = np.random.randint(1, cfg.K + 1)
        tval = k / cfg.K

        #  3. Add noise (Forward Process)
        zk = add_noise(z0, k, ab, cfg)

        # 4. Compute Score and Divergence
        z = torch.tensor(zk, device=device, requires_grad=True).float()
        t = torch.full((cfg.batch_size, 1), float(tval), device=device).float()

        opt.zero_grad()
        s = model(z, t)
        div = divergence_hutchinson(s, z)

        # 5. Hyvärinen Loss
        norm_sq = 0.5 * torch.sum(s**2, dim=1)
        loss = (norm_sq + div).mean()
        
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        bar.set_description(f"[train] loss={loss.item():.4f}")
        bar.update(1)

    bar.close()
    
    #  Plot training curve
    plt.figure()
    plt.plot(losses)
    plt.title("Latent Score Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(cfg.outdir, "latent_loss.png"))
    plt.close()


def save_grid(imgs: torch.Tensor, path: str, title: str | None = None, nrow: int = 8):
    imgs = imgs.detach().cpu().clamp(0, 1)
    n = imgs.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    
    # Handle single row case
    if ncol == 1:
        axes = np.array(axes).reshape(1, -1)
    else:
        axes = np.array(axes).reshape(ncol, nrow)
        
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            axes[r, c].axis("off")
            if idx < n:
                axes[r, c].imshow(imgs[idx, 0], cmap="gray")
            idx += 1
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    cfg = LatentConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print(f"Loading VAE from {cfg.vae_weights}...")
    vae = VAE(
        input_dim=28 * 28, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim
    ).to(device)
    
    if not os.path.exists(cfg.vae_weights):
        print(f"Error: Weights not found at {cfg.vae_weights}. Run P3_1_vae_template.py first.")
        return

    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=cfg.outdir, train=True, download=True, transform=tfm)
    loader = DataLoader(train_ds, batch_size=256, shuffle=False)

    print("Encoding MNIST to latent space...")
    Z, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            mu, _ = vae.enc(x_flat) # Use mean as the code
            Z.append(mu.cpu().numpy())
            Y.append(y.numpy())

    Zs = None
    z_mean, z_std = 0, 1

    if len(Z) > 0:
        Z = np.concatenate(Z, axis=0).astype(np.float32)
        Y = np.concatenate(Y, axis=0).astype(np.int64)
        print("Latent dataset shape:", Z.shape)

        # Standardize latents 
        z_mean = Z.mean(axis=0, keepdims=True)
        z_std = Z.std(axis=0, keepdims=True) + 1e-6
        Zs = (Z - z_mean) / z_std


        if cfg.latent_dim == 2:
            plt.figure(figsize=(6, 6))
            plt.scatter(Zs[:, 0], Zs[:, 1], s=2, c=Y, cmap="tab10", alpha=0.6)
            plt.colorbar()
            plt.title("Standardized Latent Space z=mu(x)")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.outdir, "latent_scatter.png"), dpi=200)
            plt.close()



    model = ScoreNet(hidden=cfg.score_hidden).to(device)

    print("Training Score Network on Latents...")
    train_score(Zs, cfg, model, device)

    def score_learned(z_np: np.ndarray, k: int) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            zt = torch.tensor(z_np, device=device).float()
            tt = torch.full((z_np.shape[0], 1), float(k / cfg.K), device=device).float()
            return model(zt, tt).cpu().numpy().astype(np.float32)

    print("Sampling with Reverse Diffusion...")
    # Start from N(0, I) in standardized space
    zK = np.random.randn(64, cfg.latent_dim).astype(np.float32)
    
    # Run reverse Euler integration
    zs_traj = reverse_euler(cfg, zK, score_learned)
    z0_gen_std = zs_traj[-1]

    # Un-standardize back to VAE's latent scale
    z0_gen = z0_gen_std * z_std + z_mean

    # Decode to images
    with torch.no_grad():
        z_in = torch.tensor(z0_gen, device=device).float()
        logits = vae.dec(z_in)
        x_gen = torch.sigmoid(logits).view(-1, 1, 28, 28)

    save_grid(x_gen, os.path.join(cfg.outdir, "latent_diffusion_samples.png"), 
              title="Latent Diffusion Samples")

    print("Generating VAE-only samples for comparison...")
    with torch.no_grad():
        # VAE assumes prior N(0, I)
        z_prior = torch.randn(64, cfg.latent_dim).to(device)
        logits_vae = vae.dec(z_prior)
        x_vae = torch.sigmoid(logits_vae).view(-1, 1, 28, 28)

    save_grid(x_vae, os.path.join(cfg.outdir, "vae_prior_samples.png"),
              title="Standard VAE Prior Samples")

    # Save weights
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "latent_score_net.pt"))
    print("Saved weights:", os.path.join(cfg.outdir, "latent_score_net.pt"))


if __name__ == "__main__":
    main()