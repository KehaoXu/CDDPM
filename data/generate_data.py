import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from PIL import Image

def generate_samples(
    model,
    num_samples,
    label_ratio = (0.5, 0.5),
    batch_size = 16,
    size = 128,
    save_dir="results/generated_imgs"
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    label_emb = nn.Embedding(2, 128).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    os.makedirs(save_dir, exist_ok=True) # create folder

    # all_imgs = []
    # all_labels = []

    # 根据比例生成标签
    labels_list = []
    for label, ratio in enumerate(label_ratio):
        count = int(num_samples * ratio)
        labels_list.extend([label] * count)

    # 补齐数量（如由于 int 向下取整）
    while len(labels_list) < num_samples:
        labels_list.append(0)  # 补第一个类

    labels = torch.tensor(labels_list, device=device, dtype=torch.long)
    labels = labels[torch.randperm(len(labels))] # 随机打乱

    global_idx = 0

    for i in range(0, num_samples, batch_size):
        batch_labels = labels[i:i+batch_size].to(device)
        x_gen = torch.randn((len(batch_labels), 1, size, size), device=device)

        encoder_hidden_states = label_emb(batch_labels).unsqueeze(1)  # [B, 1, 128]

        for t in tqdm(reversed(range(noise_scheduler.config.num_train_timesteps))):
            t_tensor = torch.full((len(batch_labels),), t, device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(x_gen, t_tensor, encoder_hidden_states).sample

            x_gen = noise_scheduler.step(noise_pred, t, x_gen).prev_sample

        x_gen = (x_gen.clamp(-1, 1) + 1) / 2  # 反归一化为 [0,1]

        for j in range(x_gen.size(0)):
            img = x_gen[j, 0].cpu().numpy()  # shape: [H, W]
            label = batch_labels[j].item()
            path = os.path.join(save_dir, f"sample_{global_idx:04d}_label{label}.png")
            plt.imsave(path, img, cmap='gray')
            global_idx += 1

        # all_imgs.append(x_gen.cpu())
        # all_labels.append(batch_labels.cpu())

    # return torch.cat(all_imgs, dim=0), np.concatenate(all_labels, axis=0).reshape(-1, 1)

class GenDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.transform = transforms.Compose([
        transforms.ToTensor(),                # 转成Tensor
        transforms.Normalize(mean=[.5], std=[.5])  # 归一化到[-1,1]
        ])
        self.samples = []

        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".png") and "label" in fname:
                label = int(fname.split("label")[-1].split(".")[0])
                path = os.path.join(folder, fname)
                self.samples.append((path, np.array([label])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # 灰度图
        if self.transform:
            img = self.transform(img)
        return img, label

