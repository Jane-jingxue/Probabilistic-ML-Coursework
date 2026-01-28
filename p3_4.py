from __future__ import annotations
import os
import math
from dataclasses import dataclass
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
        divergence_hutchinson,
        reverse_euler,
        alpha_bar,
    )
except ImportError:
    print("Error: P3_1_vae_template.py or P3_2_spiral_score_template.py not found.")
    exit(1)


@dataclass
class CondConfig:
    vae_weights: str = "outputs_q1_vae/vae_mnist.pt"
    
    # ----------------------------------------
    # CONFIGURATION
    # ----------------------------------------
    latent_dim: int = 2     
    hidden_dim: int = 400
    
    # Brute Force Capacity
    score_hidden: int = 512  
    embed_dim: int = 256     
    num_classes: int = 11    

    # Diffusion
    beta0: float = 0.1
    beta1: float = 12.0      
    K: int = 500             

    # Training
    steps: int = 20000      
    batch_size: int = 256
    lr: float = 1e-4        
    

    p_uncond: float = 0.2    
    guidance_w: float = 7.5  

    outdir: str = "outputs_P3.4_conditional"
    seed: int = 999          


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConditionalScoreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # Input: z + t + y_embedding
        total_input_dim = input_dim + 1 + embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.class_embed(y)
        zt_y = torch.cat([z, t, y_emb], dim=1)
        return self.net(zt_y)


def add_noise(z0, k, ab, cfg):
    ab_k = ab[k]
    eps = np.random.randn(*z0.shape).astype(np.float32)
    zk = math.sqrt(ab_k) * z0 + math.sqrt(1 - ab_k) * eps
    return zk


def train_conditional_score(Zs, Ys, cfg: CondConfig, model: nn.Module, device: str):
    ab = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    
    losses = []
    bar = tqdm(range(1, cfg.steps + 1))
    
    for step in range(1, cfg.steps + 1):
        # 1. Sample
        idx = np.random.randint(0, Zs.shape[0], size=cfg.batch_size)
        z0 = Zs[idx]
        y_np = Ys[idx]

        # 2. CFG Dropout
        mask = np.random.rand(cfg.batch_size) > cfg.p_uncond
        y_train = np.where(mask, y_np, 10) 

        # 3. Noise
        k = np.random.randint(1, cfg.K + 1)
        tval = k / cfg.K
        zk = add_noise(z0, k, ab, cfg)
        
        # 4. Tensor
        z = torch.tensor(zk, device=device, requires_grad=True).float()
        t = torch.full((cfg.batch_size, 1), float(tval), device=device).float()
        y = torch.tensor(y_train, device=device).long()

        opt.zero_grad()
        s = model(z, t, y)
        div = divergence_hutchinson(s, z)
        norm_sq = 0.5 * torch.sum(s**2, dim=1)
        loss = (norm_sq + div).mean()
        
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        bar.set_description(f"Loss: {loss.item():.4f}")
        bar.update(1)
        
    bar.close()
    
    plt.figure()
    plt.plot(losses)
    plt.title("Conditional Score Training Loss (CFG)")
    plt.savefig(os.path.join(cfg.outdir, "loss_conditional.png"))
    plt.close()


def save_grid(imgs: torch.Tensor, path: str, nrow: int = 10):
    imgs = imgs.detach().cpu().clamp(0, 1)
    n = imgs.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    if ncol * nrow > 1:
        axes = np.array(axes).flatten()
    else:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i, 0], cmap="gray")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    cfg = CondConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1. Load VAE
    vae = VAE(28 * 28, cfg.hidden_dim, cfg.latent_dim).to(device)
    if not os.path.exists(cfg.vae_weights):
        print(f"Weights not found: {cfg.vae_weights}. Run P3.1 first.")
        return
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()

    # 2. Data
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.outdir, train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    
    
    # DEBUG: VAE RECONSTRUCTION CHECK
    # This checks if the VAE itself is capable of drawing a 4 or 5
    print("Running VAE Sanity Check...")
    # Find indices for a 4 and a 5
    indices_4 = [i for i, (x, y) in enumerate(ds) if y == 4][:5]
    indices_5 = [i for i, (x, y) in enumerate(ds) if y == 5][:5]
    
    debug_imgs = []
    for idx in indices_4 + indices_5:
        x, y = ds[idx]
        x = x.to(device).view(1, -1)
        with torch.no_grad():
            mu, _ = vae.enc(x)
            x_recon = torch.sigmoid(vae.dec(mu)).view(1, 1, 28, 28)
        debug_imgs.append(x_recon)
    
    debug_grid = torch.cat(debug_imgs, dim=0)
    save_grid(debug_grid, os.path.join(cfg.outdir, "vae_debug_recons.png"), nrow=5)
    print("Saved 'vae_debug_recons.png'. PLEASE CHECK THIS FILE.")
    print("If the 4s look like 9s here, THE PROBLEM IS YOUR VAE, NOT THE DIFFUSION.")

    # 3. Encode
    Z, Y = [], []
    print("Encoding dataset...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            mu, _ = vae.enc(x)
            Z.append(mu.cpu().numpy())
            Y.append(y.numpy())
            
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    z_mean = Z.mean(axis=0, keepdims=True)
    z_std = Z.std(axis=0, keepdims=True) + 1e-6
    Zs = (Z - z_mean) / z_std

    # 4. Train
    model = ConditionalScoreNet(cfg.latent_dim, cfg.score_hidden, cfg.num_classes, cfg.embed_dim).to(device)
    print("Training Conditional Score Model (CFG)...")
    train_conditional_score(Zs, Y, cfg, model, device)

    # 5. Generate
    print("Generating digits 0-9 using Guidance...")
    all_z_gen = []
    
    for digit in range(10):
        def cfg_score_wrapper(z_np, k):
            model.eval()
            with torch.no_grad():
                B = z_np.shape[0]
                zt = torch.tensor(z_np, device=device).float()
                tt = torch.full((B, 1), float(k / cfg.K), device=device).float()
                
                y_cond = torch.full((B,), digit, device=device).long()
                y_uncond = torch.full((B,), 10, device=device).long()
                
                s_cond = model(zt, tt, y_cond)
                s_uncond = model(zt, tt, y_uncond)
                
                # Apply High Guidance
                s_final = s_uncond + cfg.guidance_w * (s_cond - s_uncond)
                
                return s_final.cpu().numpy()

        zK = np.random.randn(10, cfg.latent_dim).astype(np.float32)
        traj = reverse_euler(cfg, zK, cfg_score_wrapper)
        z0 = traj[-1]
        all_z_gen.append(z0)

    z_gen_all = np.concatenate(all_z_gen, axis=0)
    z_gen_all = z_gen_all * z_std + z_mean
    
    with torch.no_grad():
        z_in = torch.tensor(z_gen_all, device=device).float()
        x_gen = torch.sigmoid(vae.dec(z_in)).view(-1, 1, 28, 28)
        
    save_grid(x_gen, os.path.join(cfg.outdir, "conditional_grid_0to9.png"), nrow=10)
    print("Saved grid.")
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "cond_score_net.pt"))

if __name__ == "__main__":
    main()