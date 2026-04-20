import numpy as np
import torch
from detect import detect_stars
from generator import StarTrainingGenerator
from model import UNet
from scipy.optimize import linear_sum_assignment


def match_stars(gt_stars, pred_stars, max_dist=3.0):
    """
    Hungarian matching between ground truth and predicted stars.

    Returns TP, FP, FN.
    """
    if len(gt_stars) == 0 and len(pred_stars) == 0:
        return 0, 0, 0
    if len(gt_stars) == 0:
        return 0, len(pred_stars), 0  # all pred = false
    if len(pred_stars) == 0:
        return 0, 0, len(gt_stars)  # all ground truth are missed

    # dist matrix
    dist = np.zeros((len(gt_stars), len(pred_stars)))
    for i, gt in enumerate(gt_stars):
        for j, pred in enumerate(pred_stars):
            d = np.hypot(
                gt["x"] - pred["x"], gt["y"] - pred["y"]
            )  # euclid dist.
            dist[i, j] = d if d <= max_dist else 1e9

    row_ind, col_ind = linear_sum_assignment(
        dist
    )  # finds one to one matching that minimizes sum of distances
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= max_dist:
            tp += 1
    fp = len(pred_stars) - tp  # pred not matched as tp
    fn = len(gt_stars) - tp  # ground truth not matched as tp
    return tp, fp, fn


def evaluate_model(model, generator, num_test=500, device="cpu"):
    model.eval()
    total_tp = total_fp = total_fn = 0
    for _ in range(num_test):
        n_stars = np.random.randint(5, 30)
        img, gt_stars, _, _ = generator.generate(n_stars)
        img_t = (
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)
        )
        with torch.no_grad():
            heat, mag = model(img_t)
        heat_np = heat.squeeze().cpu().numpy()
        mag_np = mag.squeeze().cpu().numpy()
        pred_stars = detect_stars(heat_np, mag_np, peak_threshold=0.4)
        tp, fp, fn = match_stars(gt_stars, pred_stars)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"Test on {num_test} images:")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    return precision, recall, f1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(
        torch.load("best_star_model.pth", map_location=device)
    )
    gen = StarTrainingGenerator(
        size=128, seed=123
    )  # same parameters as training
    evaluate_model(model, gen, num_test=200, device=device)
