import sys
from datetime import date
from os import wait
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from test_cv2 import get_debayered_white


def get_sources(data, threshold_sigma, fwhm):
    _, med, std = sigma_clipped_stats(data)
    threshold = threshold_sigma * std
    sources = DAOStarFinder(threshold=threshold, fwhm=fwhm)(data)
    count = len(sources) if sources is not None else 0
    return sources, count


def detect(filepath: str):
    print(f"detecting data for {filepath}")
    data = fits.getdata(filepath).astype(np.float64)
    data = get_debayered_white(data)
    print(data.ndim)
    visualize(data, [])
    thresholds = np.median(data)  # edit for better data?
    fwhms = np.arange(50, 55, 1)
    best_count = -1
    best_params = None
    best_sources = None
    for f in fwhms:
        sources, count = get_sources(data, thresholds, f)
        if count > best_count:
            best_count = count
            best_params = (thresholds, f)
            best_sources = sources

    t_best, f_best = best_params
    print(
        f"Best detection: {best_count} stars (threshold={t_best}sigma, fwhm={f_best})"
    )
    visualize(data, best_sources)

    # output_file = Path(filepath).with_suffix(f"_output_{date.today()}.txt")
    # currently not saving, but potentially save in the future to use and access from some sort of db?
    # with open(output_file, "w") as f:
    #     f.write(f"Best detection: {best_count} stars\n")
    #     f.write(f"Threshold: {t_best}σ, FWHM: {f_best}\n")
    #     if best_sources is not None:
    #         f.write(str(best_sources))
    print(f"Best detection: {best_count} stars\n")
    print(f"Threshold: {t_best}σ, FWHM: {f_best}\n")
    if best_sources is not None:
        print(str(best_sources))


def visualize(data, sources):
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray_r")

    if sources is not None:

        for source in sources:
            circle = patches.Circle(
                (source["xcentroid"], source["ycentroid"]),
                5,
                edgecolor="r",
                facecolor="none",
                linewidth=1.5,
            )
            ax.add_patch(circle)
    plt.show()


if __name__ == "__main__":
    for i in sys.argv[1:]:
        detect(i)
