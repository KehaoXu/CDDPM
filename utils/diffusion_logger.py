import os
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, train_losses, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_losses": train_losses,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at: {path}")

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] # last epoch
    train_losses = checkpoint["train_losses"]
    print(f"Checkpoint loaded from: {path} | Resume from epoch {epoch}")
    return epoch + 1, train_losses

def plot_ddpm_training_loss(model, resume_path="results/diffusion_checkpoints/latest.pt", save_path="results"):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    if not os.path.exists(resume_path):
        print(f"No checkpoint found at {resume_path}.")
        return

    _, train_losses = load_checkpoint(resume_path, model, optimizer)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss over Epoch")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.savefig(os.path.join(save_path, "ddpm_loss.png"))