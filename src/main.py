import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def main():
    for i in sys.argv[1:]:
        print(i)
        open_fits_file(i)


def detect_stars(image_data, blur_strength, detection_threshold=0.05):
    blurred_image = gaussian_filter(image_data, sigma=blur_strength)
    normalized_image = (blurred_image - np.min(blurred_image)) / (
        np.max(blurred_image) - np.min(blurred_image)
    )
    binary_star_map = (normalized_image > detection_threshold).astype(np.uint8)
    total_labels, _, _, _ = cv2.connectedComponentsWithStats(
        binary_star_map, connectivity=8
    )
    star_count = total_labels - 1
    return star_count, normalized_image, binary_star_map


def load_fits_image(fn: str):
    with fits.open(fn) as file:
        data = file[0].data
        return data


def open_fits_file(fn: str):
    data = load_fits_image(fn)
    blur = [0.1, 0.5, 1, 2]
    stars = []
    images = []
    for s in blur:
        count, normalized, image = detect_stars(data, s)
        stars.append(count)
        images.append(image)
        print(f"blur {s} detected {count} stars")

    plt.figure(figsize=(12, 8))
    for i, s in enumerate(blur):
        _ = plt.subplot(2, len(blur), i + 1)
        _ = plt.imshow(images[i], cmap="gray")
        _ = plt.title(f"Blur: {s}")
        _ = plt.axis("off")

    _ = plt.subplot(2, 1, 2)
    _ = plt.plot(blur, stars, marker="o", linestyle="-")
    _ = plt.xlabel("gaussian blur strength (\\sigma)")
    _ = plt.ylabel("# of stars detected")
    _ = plt.title("star detection across blur levels")
    _ = plt.tight_layout()
    _ = plt.show()


if __name__ == "__main__":
    main()
