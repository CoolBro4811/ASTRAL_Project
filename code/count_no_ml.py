import numpy as np
from astropy.io import fits
from scipy.ndimage import label, center_of_mass, maximum_filter
import sys


def find_stars(image, threshold_sigma=5, min_separation=3):
    """
    Find stars in an image using simple peak detection.

    Parameters:
    - image: 2D numpy array (already bias/dark/flat corrected)
    - threshold_sigma: number of standard deviations above background to consider a peak
    - min_separation: minimum pixel distance between stars (for non‑max suppression)

    Returns:
    - list of (x, y) centroids (subpixel)
    """
    # Estimate background noise
    background = np.median(image)
    noise = np.std(
        image[image < background + 3 * background]
    )  # robust estimate

    # Threshold
    threshold = background + threshold_sigma * noise
    binary = image > threshold

    # Label connected components
    labeled, num_features = label(binary)

    centroids = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        # Use flux‑weighted centroid for subpixel accuracy
        region_flux = image[y, x]
        if region_flux.sum() > 0:
            cx = np.average(x, weights=region_flux)
            cy = np.average(y, weights=region_flux)
        else:
            cy, cx = center_of_mass(labeled == i)
        centroids.append((cx, cy))

    # Optional: remove duplicates / enforce min separation
    # (simple greedy filter)
    if min_separation > 0:
        filtered = []
        for c in sorted(
            centroids, key=lambda p: image[int(p[1]), int(p[0])], reverse=True
        ):
            if all(
                np.hypot(c[0] - f[0], c[1] - f[1]) >= min_separation
                for f in filtered
            ):
                filtered.append(c)
        centroids = filtered

    return centroids


def count_stars_in_fits(fits_path, threshold_sigma=5):
    """Load FITS, detect stars, return count and centroids."""
    img = fits.getdata(fits_path).astype(np.float32)
    # If the image is already calibrated (bias/dark/flat subtracted),
    # we can proceed directly. Optionally clip negative values.
    img = np.clip(img, 0, None)
    centroids = find_stars(img, threshold_sigma=threshold_sigma)
    return len(centroids), centroids


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python count_stars_classical.py <fits_file> [threshold_sigma]"
        )
        sys.exit(1)
    fits_file = sys.argv[1]
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    nstars, cents = count_stars_in_fits(fits_file, threshold_sigma=sigma)
    print(f"Number of stars detected (classical method): {nstars}")
    for i, (x, y) in enumerate(cents[:5]):
        print(f"  Star {i+1}: ({x:.2f}, {y:.2f})")
    if nstars > 5:
        print(f"  ... and {nstars-5} more")
