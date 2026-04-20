import numpy as np
import torch
from scipy.ndimage import label, maximum_filter


def detect_stars(heatmap, magmap, peak_threshold=0.5, min_distance=3):
    """
    convert predicted heatmap and magnitude map to star positions and magnitudes

    Args:
        heatmap: 2D numpy array (H,W) in [0,1]
        magmap: 2D numpy array (H,W)
        peak_threshold: minimum heatmap value to consider a star
        min_distance: minimum pixel separation between peaks (uses max filter)

    Returns:
        list of dicts: [{'x': float, 'y': float, 'mag': float}, ...]
    """
    # find local maxima using maximum filter
    footprint = np.ones((min_distance, min_distance))
    max_filtered = maximum_filter(heatmap, footprint=footprint)
    maxima = (heatmap == max_filtered) & (heatmap >= peak_threshold)

    # connected components (to avoid duplicates)
    labeled, num_features = label(maxima)
    stars = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        # pixel with highest heatmap value within the component as center
        idx = np.argmax(heatmap[y, x])
        cy, cx = y[idx], x[idx]
        mag = magmap[cy, cx]
        stars.append({"x": float(cx), "y": float(cy), "mag": float(mag)})
    return stars
