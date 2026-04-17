"""
SRCNN Training Script
=====================
Train the SRCNN model on X-ray or general medical images.
After training, the weights are saved as 'srcnn_weights.pth'
and loaded automatically by app.py.

Usage:
    python train_srcnn.py --epochs 100 --data_dir ./training_images

References:
    - Paper: https://arxiv.org/abs/1501.00092
    - Training data: DIV2K, Set5, Set14, or medical image datasets
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class XRayDataset(Dataset):
    """
    Dataset that:
    1. Loads high-res images from data_dir
    2. Downscales to create low-res (input)
    3. Returns (low-res bicubic upscaled, high-res) pairs
    """
    def __init__(self, data_dir: str, patch_size: int = 33, scale: int = 2):
        self.patch_size = patch_size
        self.scale = scale
        self.image_paths = list(Path(data_dir).glob('**/*.jpg')) + \
                           list(Path(data_dir).glob('**/*.png'))
        print(f"Found {len(self.image_paths)} training images")

    def __len__(self):
        return len(self.image_paths) * 10  # 10 patches per image

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        img = cv2.imread(str(self.image_paths[img_idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        img = img.astype(np.float32) / 255.0
        h, w = img.shape

        # Random crop
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
            hr_patch = img[top:top + self.patch_size, left:left + self.patch_size]
        else:
            hr_patch = cv2.resize(img, (self.patch_size, self.patch_size))

        # Create LR by downscaling then upscaling (bicubic)
        lr_small = cv2.resize(
            hr_patch,
            (self.patch_size // self.scale, self.patch_size // self.scale),
            interpolation=cv2.INTER_CUBIC
        )
        lr_patch = cv2.resize(
            lr_small,
            (self.patch_size, self.patch_size),
            interpolation=cv2.INTER_CUBIC
        )

        lr_tensor = torch.FloatTensor(lr_patch).unsqueeze(0)
        hr_tensor = torch.FloatTensor(hr_patch).unsqueeze(0)
        return lr_tensor, hr_tensor


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    dataset = XRayDataset(args.data_dir, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print(f"\n📚 Training SRCNN for {args.epochs} epochs...")
    print(f"   Dataset: {len(dataset)} patches")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}\n")

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'srcnn_weights.pth')
            print(f"  ✅ Saved best model (loss={best_loss:.6f})")

    print(f"\n🎉 Training complete! Best loss: {best_loss:.6f}")
    print("   Weights saved to: srcnn_weights.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SRCNN')
    parser.add_argument('--data_dir', type=str, default='./training_images',
                        help='Directory with training images')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=33)
    args = parser.parse_args()
    train(args)
