import numpy as np
import torch
from generator import StarTrainingGenerator
from model import UNet
from scipy.ndimage import label, maximum_filter


def extract_star_centroids(heatmap, threshold=0.3, min_distance=3):
    """Extract centroids from a heatmap using local maxima."""
    footprint = np.ones((min_distance, min_distance))
    max_filtered = maximum_filter(heatmap, footprint=footprint)
    maxima = (heatmap == max_filtered) & (heatmap >= threshold)
    labeled, num_features = label(maxima)
    centroids = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        # Use the maximum pixel position (or weighted centroid)
        idx = np.argmax(heatmap[y, x])
        centroids.append((x[idx], y[idx]))
    return centroids


def evaluate_model(
    model, generator, num_test=200, device="cpu", peak_threshold=0.3
):
    model.eval()
    total_tp = total_fp = total_fn = 0
    for _ in range(num_test):
        img, gt_stars, _, _ = generator.generate(
            num_stars=np.random.randint(5, 20)
        )
        img_t = (
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
        )
        with torch.no_grad():
            heat, mag = model(img_t)
        heat_np = heat.squeeze().cpu().numpy()
        # Ground truth centroids from the true heatmap? Simpler: use the generator's labels
        true_centroids = [(s["x"], s["y"]) for s in gt_stars]
        pred_centroids = extract_star_centroids(
            heat_np, threshold=peak_threshold
        )
        # Matching (Hungarian) – same as in training
        from scipy.optimize import linear_sum_assignment

        if len(true_centroids) == 0 and len(pred_centroids) == 0:
            continue
        if len(true_centroids) == 0:
            total_fp += len(pred_centroids)
            continue
        if len(pred_centroids) == 0:
            total_fn += len(true_centroids)
            continue
        dist = np.zeros((len(true_centroids), len(pred_centroids)))
        for i, (gtx, gty) in enumerate(true_centroids):
            for j, (px, py) in enumerate(pred_centroids):
                d = np.hypot(gtx - px, gty - py)
                dist[i, j] = d if d <= 3.0 else 1e9
        row_ind, col_ind = linear_sum_assignment(dist)
        tp = sum(1 for r, c in zip(row_ind, col_ind) if dist[r, c] <= 3.0)
        fp = len(pred_centroids) - tp
        fn = len(true_centroids) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"Test on {num_test} images (threshold={peak_threshold}):")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(
        torch.load("best_star_model.pth", map_location=device)
    )
    gen = StarTrainingGenerator(
        size=96,
        star_flux=800,
        background=5,
        read_noise=1.5,
        target_sigma=2.0,
        seed=42,
    )
    # Try different thresholds to find the best F1
    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
        evaluate_model(
            model, gen, num_test=200, device=device, peak_threshold=thresh
        )
