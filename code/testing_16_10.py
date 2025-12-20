import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import center_of_mass, find_objects, label
from skimage import measure


def get_components(data):
    print(data)
    data = -data
    print(data)
    mean = np.percentile(data, 95)
    mask = data > mean
    labels, num = label(mask)

    print(num, "regions found")

    regions = []
    for i in range(1, num + 1):
        region_mask = labels == i
        coords = np.argwhere(region_mask)
        bbox = find_objects(labels)[i - 1]
        centroid = center_of_mass(region_mask)
        area = np.sum(region_mask)

        if area < 68:
            continue
        regions.append(
            {
                "id": i,
                "bbox": bbox,
                "centroid": centroid,
                "area": area,
                "coords": coords,
            }
        )

    return labels, regions


def visualize(data, labels, regions, title="Detected Regions"):
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap="gray_r", origin="lower")

    for r in regions:
        region_mask = labels == r["id"]
        contours = measure.find_contours(region_mask, 0.5)
        for contour in contours:
            plt.plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=1.5,
                label=f"Region {r['id']}",
            )

    plt.title(title)
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.show()


def main(files: list[str]):
    for file in files:
        with fits.open(file) as f:
            data = np.array(f[0].data)
        labels, regions = get_components(data)

        print(f"\nFile: {file}, found {len(regions)} components")
        for r in regions:
            print(
                f"  Region {r['id']}: area={r['area']}, centroid={r['centroid']}"
            )

        visualize(data, labels, regions, title=f"Regions in {file}")


if __name__ == "__main__":
    main(sys.argv[1:])
