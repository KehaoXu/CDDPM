import os
import numpy as np
import torch
from torch import nn, optim
from diffusers import DDPMScheduler
from tqdm import tqdm

from utils.diffusion_logger import save_checkpoint, load_checkpoint

def train_ddpm(
    model,
    train_loader,
    epochs = 10,
    save_every = 10,
    lr = 1e-4,
    checkpoint_dir = "results/diffusion_checkpoints",
    resume_file = "latest.pt",
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    resume_path = os.path.join(checkpoint_dir, resume_file)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    label_emb = nn.Embedding(2, 128).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    train_losses = []

    os.makedirs(checkpoint_dir, exist_ok=True)
    if resume_path and os.path.exists(resume_path):
        start_epoch, train_losses = load_checkpoint(resume_path, model, optimizer)
    else:
        start_epoch = 1
    
    if epochs == 0:
        return

    # train_losses = np.ones(epochs)       # 每个 epoch 的平均 loss

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        batch_losses = []       # 当前 epoch 所有 batch 的 loss
        pbar = tqdm(train_loader)

        for images, labels in pbar:  # 正确方式：直接解包 tuple
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.size(0),), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            encoder_hidden_states = label_emb(labels).unsqueeze(1)  # [B, 1, 128]

            noise_pred = model(noisy_images, timesteps, encoder_hidden_states).sample

            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            pbar.set_description(f"DDPM Epoch {epoch} loss: {np.mean(batch_losses):.4f}")

        train_losses.append(np.mean(batch_losses)) # epoch loss

        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt"))

    save_checkpoint(model, optimizer, epoch, train_losses, resume_path)  # always keep latest


