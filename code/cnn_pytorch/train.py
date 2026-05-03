from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from generator import StarTrainingGenerator
from model import UNet
from scipy.ndimage import label, maximum_filter
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def custom_collate(batch):
    imgs, heats, mags, stars = zip(*batch)

    imgs = torch.stack(imgs)
    heats = torch.stack(heats)
    mags = torch.stack(mags)

    # keep stars as list (DO NOT stack)
    return imgs, heats, mags, list(stars)


class StarDataset(Dataset):
    def __init__(
        self,
        generator,
        num_samples,
        stars_per_image_range=(5, 30),
        mag_range=(5, 12),
        normalize=True,
    ):
        self.gen = generator
        self.num_samples = num_samples
        self.stars_range = stars_per_image_range
        self.mag_range = mag_range
        self.normalize = normalize

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n_stars = np.random.randint(
            self.stars_range[0], self.stars_range[1] + 1
        )
        img, stars, heatmap, magmap = self.gen.generate(
            n_stars, mag_range=self.mag_range
        )
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            heatmap = np.fliplr(heatmap).copy()
            magmap = np.fliplr(magmap).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            heatmap = np.flipud(heatmap).copy()
            magmap = np.flipud(magmap).copy()

        if self.normalize:
            img = np.log1p(np.maximum(img, 0))
            img = (img - img.mean()) / (img.std() + 1e-6)

        img_t = torch.from_numpy(img).unsqueeze(0).float()
        heat_t = torch.from_numpy(heatmap).unsqueeze(0).float()
        mag_t = torch.from_numpy(magmap).unsqueeze(0).float()
        return img_t, heat_t, mag_t, stars


def loss_fn(
    pred_heat_logits,
    true_heat,
    pred_mag,
    true_mag,
    heat_weight=1.0,
    mag_weight=0.2,
):

    heat_loss = F.binary_cross_entropy_with_logits(pred_heat_logits, true_heat)

    pred_heat = torch.sigmoid(pred_heat_logits)

    mask = (true_heat > 0.05).float()

    if mask.sum() > 0:
        mag_loss = F.smooth_l1_loss(
            pred_mag * mask, true_mag * mask, reduction="sum"
        ) / mask.sum().clamp_min(1.0)
    else:
        mag_loss = torch.tensor(0.0, device=pred_mag.device)

    return heat_weight * heat_loss + mag_weight * mag_loss


def match_stars(gt_stars, pred_stars, max_dist=3.0):
    if len(gt_stars) == 0 and len(pred_stars) == 0:
        return 0, 0, 0
    if len(gt_stars) == 0:
        return 0, len(pred_stars), 0
    if len(pred_stars) == 0:
        return 0, 0, len(gt_stars)

    dist = np.zeros((len(gt_stars), len(pred_stars)))
    for i, gt in enumerate(gt_stars):
        for j, pred in enumerate(pred_stars):
            # both are formatted as (x, y) pairs
            d = np.hypot(gt[0] - pred[0], gt[1] - pred[1])
            dist[i, j] = d if d <= max_dist else 1e9

    row_ind, col_ind = linear_sum_assignment(dist)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if dist[r, c] <= max_dist)
    fp = len(pred_stars) - tp
    fn = len(gt_stars) - tp
    return tp, fp, fn


def detect_stars(heatmap, threshold=0.3, min_distance=3):
    heatmap = heatmap.squeeze()

    footprint = np.ones((min_distance, min_distance))
    max_filtered = maximum_filter(heatmap, footprint=footprint)

    maxima = (heatmap == max_filtered) & (heatmap >= threshold)

    labeled, num = label(maxima)

    stars = []
    for i in range(1, num + 1):
        y, x = np.where(labeled == i)
        idx = np.argmax(heatmap[y, x])
        stars.append((x[idx], y[idx]))

    return stars


def extract_star_centroids(heatmap, threshold=0.5, min_distance=3):
    # If heatmap is 3D (batch,1,H,W) or (1,H,W), squeeze it
    if heatmap.ndim == 3:
        heatmap = heatmap.squeeze(0)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")
    from scipy.ndimage import label, maximum_filter

    footprint = np.ones((min_distance, min_distance))
    max_filtered = maximum_filter(heatmap, footprint=footprint)
    maxima = (heatmap == max_filtered) & (heatmap >= threshold)
    labeled, num_features = label(maxima)
    centroids = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        # Take the pixel with highest heatmap value (or centroid of region)
        idx = np.argmax(heatmap[y, x])
        centroids.append((x[idx], y[idx]))
    return centroids


def evaluate(model, loader, device):
    model.eval()

    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for img, heat, mag, stars in loader:
            img = img.to(device)

            pred_heat_logits, _ = model(img)
            pred_heat = torch.sigmoid(pred_heat_logits).cpu().numpy()

            for b in range(img.size(0)):
                pred = detect_stars(pred_heat[b, 0])

                gt = [(s["x"], s["y"]) for s in stars[b]]
                # print(gt)

                tp, fp, fn = match_stars(gt, pred)

                total_tp += tp
                total_fp += fp
                total_fn += fn

    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return prec, rec, f1


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for img, heat, mag, _ in tqdm(loader, desc="Training"):
        img, heat, mag = img.to(device), heat.to(device), mag.to(device)

        optimizer.zero_grad()

        with autocast():
            pred_heat_logits, pred_mag = model(img)
            loss = loss_fn(pred_heat_logits, heat, pred_mag, mag)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_size = 96
    batch_size = 64
    epochs = 40

    train_gen = StarTrainingGenerator(size=image_size, seed=67)
    val_gen = StarTrainingGenerator(size=image_size, seed=1067)

    train_dataset = StarDataset(train_gen, 5000, (5, 20))
    val_dataset = StarDataset(val_gen, 500, (5, 20))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate,
    )

    model = UNet(in_channels=1, out_channels=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = GradScaler()

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler)

        prec, rec, f1 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d} | Loss {loss:.4f} | "
            f"P {prec:.3f} R {rec:.3f} F1 {f1:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> saved")

    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()
