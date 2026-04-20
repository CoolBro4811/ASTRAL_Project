from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from generator import StarTrainingGenerator
from model import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class StarDataset(Dataset):
    """dataset using the generator"""

    def __init__(
        self,
        generator,
        num_samples,
        stars_per_image_range=(5, 30),
        mag_range=(5, 12),
    ):
        self.gen = generator
        self.num_samples = num_samples
        self.stars_range = stars_per_image_range
        self.mag_range = mag_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n_stars = np.random.randint(
            self.stars_range[0], self.stars_range[1] + 1
        )
        img, _, heatmap, magmap = self.gen.generate(
            n_stars, mag_range=self.mag_range
        )
        # Convert to tensor (Channel, Height, Width)
        img_t = torch.from_numpy(img).unsqueeze(0).float()
        heat_t = torch.from_numpy(heatmap).unsqueeze(0).float()
        mag_t = torch.from_numpy(magmap).unsqueeze(0).float()
        return img_t, heat_t, mag_t


def heatmap_mse_loss(pred_heat, true_heat, pred_mag, true_mag, threshold=0.1):
    """MSE for heatmap + MSE for magnitude only where true_heat > threshold."""
    heat_loss = F.mse_loss(pred_heat, true_heat)
    # magnitude loss only on star pixels
    mask = (true_heat > threshold).float()
    if mask.sum() > 0:
        mag_loss = (
            F.mse_loss(pred_mag * mask, true_mag * mask, reduction="sum")
            / mask.sum()
        )
    else:
        mag_loss = torch.tensor(0.0, device=pred_mag.device)
    return heat_loss + mag_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for img, heat, mag in tqdm(loader):
        img, heat, mag = img.to(device), heat.to(device), mag.to(device)
        optimizer.zero_grad()
        pred_heat, pred_mag = model(img)
        loss = heatmap_mse_loss(pred_heat, heat, pred_mag, mag)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    """config"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 128
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4

    """generator + dataset"""
    gen = StarTrainingGenerator(
        size=image_size,
        psf_sigma=1.2,
        background=5,
        star_flux=500,
        read_noise=1.5,
        seed=42,
    )
    train_dataset = StarDataset(
        gen, num_samples=10000, stars_per_image_range=(5, 25)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_dataset = StarDataset(
        gen, num_samples=500, stars_per_image_range=(5, 25)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    """model"""
    model = UNet(in_channels=1, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # location of evaluation func
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_star_model.pth")
            print("     -> Saved best model")

    print("Training complete!")


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, heat, mag in loader:
            img, heat, mag = img.to(device), heat.to(device), mag.to(device)
            pred_heat, pred_mag = model(img)
            loss = heatmap_mse_loss(pred_heat, heat, pred_mag, mag)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
