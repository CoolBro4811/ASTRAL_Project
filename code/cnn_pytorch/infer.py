import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from detect import detect_stars
from model import UNet


def load_and_preprocess_fits(fits_path):
    """load fits file that has been preprocessed (bias/dark/flat corrected) using ../pre_analyze_improved.py"""
    data = fits.getdata(fits_path).astype(np.float32)
    # optional: normalize to [0,1] for the model
    # data = (data - data.min()) / (data.max() - data.min() + 1e-8 (for /0 error))
    return data


def run_inference(model, image, device, peak_threshold=0.5):
    """run model on a single image and return detected stars"""
    model.eval()
    # Prepare input: shape (1,1,H,W)
    # (C, H, W) from before
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        heatmap, magmap = model(img_t)
    heat_np = heatmap.squeeze().cpu().numpy()
    mag_np = magmap.squeeze().cpu().numpy()
    stars = detect_stars(heat_np, mag_np, peak_threshold=peak_threshold)
    return stars, heat_np, mag_np


def overlay_detections(image, stars, save_path=None):
    """plot image with detected star positions"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray", origin="lower")
    for star in stars:
        plt.plot(star["x"], star["y"], "r+", markersize=8, mew=2)
        plt.text(
            star["x"] + 3,
            star["y"] + 3,
            f"{star['mag']:.1f}",
            color="cyan",
            fontsize=8,
        )
    plt.title(f"Detected stars: {len(stars)}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(
        torch.load("best_star_model.pth", map_location=device)
    )

    # path to your preprocessed FITS file (output of the user's main() function)
    fits_file = sys.argv[1] if len(sys.argv) > 1 else "analyzed.fits"
    image = load_and_preprocess_fits(fits_file)

    stars, heat, mag = run_inference(model, image, device, peak_threshold=0.4)
    print(f"Detected {len(stars)} stars")
    for s in stars[:5]:
        print(f"  Star at ({s['x']:.2f}, {s['y']:.2f}) mag {s['mag']:.2f}")

    overlay_detections(image, stars, save_path="detection_result.png")
