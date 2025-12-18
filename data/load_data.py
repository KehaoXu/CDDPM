# import os
# os.environ["MEDMNIST_DATA_DIR"] = "/content/drive/MyDrive/MLSP Project/breastmnist"

from medmnist import BreastMNIST
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

def load_medmnist_dataset(size=128):

    # 定义数据增强/预处理
    data_transform = transforms.Compose([
        # transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),                # 转成Tensor
        transforms.Normalize(mean=[.5], std=[.5])  # 归一化到[-1,1]
    ])

    # 加载训练集
    train_dataset = BreastMNIST(
        split='train',
        transform=data_transform,
        download=True,
        size = size
    )

    test_dataset = BreastMNIST(
        split='test',
        transform=data_transform,
        download=True,
        size = size
    )

    return train_dataset, test_dataset

def sample_subset(dataset, total_samples=120, label_ratio=(0.5, 0.5)):
    label_counts = {0: int(total_samples * label_ratio[0]), 1: int(total_samples * label_ratio[1])}
    label_map = {0: [], 1: []}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, np.ndarray):
            label = label.item()
        else:
            label = int(label)
        if label in label_map and len(label_map[label]) < label_counts[label]:
            label_map[label].append(idx)
        if all(len(label_map[l]) == label_counts[l] for l in label_counts):
            break

    selected_indices = label_map[0] + label_map[1]
    return Subset(dataset, selected_indices)

if __name__ == "__main__":
    train_dataset_raw, test_dataset = load_medmnist_dataset()
    train_dataset = sample_subset(train_dataset_raw, total_samples = 120, label_ratio=(0.5, 0.5))

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2*16, shuffle=False)
    
    print(len(train_dataset))
    images, labels = next(iter(train_loader))
    print("image shape:", images.shape)
    print("label shape:", labels.shape)

    image = images.squeeze()
    la = labels.squeeze()

    fig, axes = plt.subplots(2, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = image[i].squeeze().cpu().numpy()           # [28, 28]
        label = int(la[i])                            # 0:bad, 1:good
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.suptitle("BreastMNIST: Sample Images with Labels", fontsize=14)
    plt.tight_layout()
    plt.show()