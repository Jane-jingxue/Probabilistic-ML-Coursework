#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 1/3: VAE for MNIST (P3.1)

This file is a minimal PyTorch skeleton that students can COMPLETE.
It includes only what students need (data loading + structure + TODOs).

Students must fill:
- Encoder/decoder architecture choices
- ELBO (negative ELBO) details
- Training curves + plots
- Generation grid

Run:
  python P3_1_vae_template.py
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class VAEConfig:
    latent_dim: int = 2        
    hidden_dim: int = 400        
    batch_size: int = 256        
    epochs: int = 20             
    lr: float = 1e-3             
    use_bce: bool = True    
    outdir: str = "outputs_q1_vae"
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    """q_phi(z|x) = N(mu(x), diag(sigma(x)^2))"""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # MLP Encoder: Input -> Hidden -> (Mu, LogVar)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.linear1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """p_theta(x|z)"""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(z))
        logits = self.linear2(x)
        return logits


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc = Encoder(input_dim, hidden_dim, latent_dim)
        self.dec = Decoder(latent_dim, hidden_dim, input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        logits = self.dec(z)
        return logits, mu, logvar


def negative_elbo(
    x: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    use_bce: bool,
):
    """
    Negative ELBO = Reconstruction Loss + KL Divergence. 
    """
    # Reconstruction term: 
    if use_bce:
        # For Bernoulli likelihood, maximizing log-likelihood is equivalent to 
        # minimizing Binary Cross Entropy. reduction='sum' sums over batch and pixels.
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    else:
        # Gaussian likelihood (assuming unit variance) 
        recon = F.mse_loss(torch.sigmoid(logits), x, reduction='sum')

    # KL Divergence term: D_KL(q(z|x) || p(z))
    # Analytic KL for two Gaussians where p(z) ~ N(0, I)
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Loss (Negative ELBO)
    total = recon + kl
    return total, recon, kl


def save_image_grid(
    x: torch.Tensor, path: str, nrow: int = 8, title: str | None = None
):
    x = x.detach().cpu().clamp(0, 1)
    n = x.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    # Handle single row case
    if ncol == 1:
        axes = axes.reshape(1, -1)
    
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            axes[r, c].axis("off")
            if idx < n:
                axes[r, c].imshow(x[idx, 0], cmap="gray")
            idx += 1
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    cfg = VAEConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Data Loading
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=cfg.outdir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=cfg.outdir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    vae = VAE(28 * 28, cfg.hidden_dim, cfg.latent_dim).to(device)
    # Optimizer 
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.lr)

    # Training Loop
    history = {'loss': [], 'recon': [], 'kl': []}
    
    print("Starting VAE training...")
    bar = tqdm(range(1, cfg.epochs + 1))
    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1) # Flatten to (batch, 784)
            
            opt.zero_grad()
            logits, mu, logvar = vae(x_flat)
            loss, recon, kl = negative_elbo(x_flat, logits, mu, logvar, cfg.use_bce)
            
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        # Average over dataset size
        N = len(train_loader.dataset)
        avg_loss = epoch_loss / N
        avg_recon = epoch_recon / N
        avg_kl = epoch_kl / N
        
        history['loss'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['kl'].append(avg_kl)

        bar.set_description(
            f"Ep {epoch} | Loss: {avg_loss:.1f} (Re: {avg_recon:.1f}, KL: {avg_kl:.1f})"
        )
        bar.update(1)

    # P3.1.3 Plot training curves 
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Neg ELBO')
    plt.plot(history['recon'], label='Reconstruction')
    plt.plot(history['kl'], label='KL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Curves')
    plt.savefig(os.path.join(cfg.outdir, "training_curves.png"))
    plt.close()

    # P3.1.4 Generation 
    print("Generating samples...")
    vae.eval()
    with torch.no_grad():
        # Sample z ~ N(0, 1)
        z_sample = torch.randn(64, cfg.latent_dim).to(device)
        
        # Decode
        logits_gen = vae.dec(z_sample)
        x_gen = torch.sigmoid(logits_gen) # Convert logits to probabilities
        
        # Reshape to image
        x_gen = x_gen.view(-1, 1, 28, 28)
        
        save_image_grid(x_gen, os.path.join(cfg.outdir, "generated_samples.png"), 
                        title="Generated Samples from Prior")
        print("Samples saved to", os.path.join(cfg.outdir, "generated_samples.png"))

    # Save model weights
    torch.save(vae.state_dict(), os.path.join(cfg.outdir, "vae_mnist.pt"))
    print("Saved weights:", os.path.join(cfg.outdir, "vae_mnist.pt"))


if __name__ == "__main__":
    main()