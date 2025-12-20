import sys
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# from astropy.table import vstack
# use for stacking multiple data - like different colors?
from photutils.detection import DAOStarFinder

counts = {}
# {[thres, fwhm]:{filepath, sources, count}}


def get_sources(filepath, data, threshold, fwhm):
    mean, med, std = sigma_clipped_stats(data)
    threshold = 3 * std
    fwhm = 3.0
    sources = DAOStarFinder(threshold=threshold, fwhm=fwhm)(data - med)
    print(f"threshold: {threshold}")
    print(f"fwhm: {fwhm}")
    # print(sources)
    print(f"{len(sources)} detected. Continuing...\n")
    counts[(threshold, fwhm)] = {
        "filepath": filepath,
        "sources": sources,
        "count": len(sources),
    }


def detect(filepath: str):
    print(f"detecting data for {filepath}")
    data = np.asarray(fits.getdata(filepath))
    # data = -data
    mean, med, std = sigma_clipped_stats(data)
    get_sources(filepath, data, med + 3.0 * std, 4)

    max_count = -1
    max_threshold = 0
    max_fwhm = 0

    min_count = sys.maxsize
    min_threshold = 0
    min_fwhm = 0

    for i in counts:
        if max_count < counts[i]["count"]:
            max_count = counts[i]["count"]
            max_threshold = i[0]
            max_fwhm = i[1]
        if min_count > counts[i]["count"]:
            min_count = counts[i]["count"]
            min_threshold = i[0]
            min_fwhm = i[1]
    print(
        f"Max detected for file: {filepath}\nThreshold:{max_threshold}\nFWHM:{max_fwhm}\n{counts[(max_threshold, max_fwhm)]}"
    )
    print(
        f"Min detected for file: {filepath}\nThreshold:{min_threshold}\nFWHM:{min_fwhm}\n{counts[(min_threshold, min_fwhm)]}"
    )
    visualize(data, max_threshold, max_fwhm)
    with open(f"./{filepath.split(".")[0][5:]}_output_{date.today()}") as file:
        file.write(
            f"Max detected for file: {filepath}\nThreshold:{max_threshold}\nFWHM:{max_fwhm}\n{counts[(max_threshold, max_fwhm)]}"
        )
        file.write(
            f"Min detected for file: {filepath}\nThreshold:{min_threshold}\nFWHM:{min_fwhm}\n{counts[(min_threshold, min_fwhm)]}"
        )


def visualize(data, max_threshold, max_fwhm):
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray_r")

    sources = counts[(max_threshold, max_fwhm)]["sources"]
    for i, source in enumerate(sources):
        circle = matplotlib.patches.Circle(
            (source["xcentroid"], source["ycentroid"]),
            10,
            edgecolor="r",
            facecolor="none",
            linewidth=1.5,
        )
        ax.add_patch(circle)
    plt.show()


if __name__ == "__main__":
    for i in sys.argv[1:]:
        detect(i)
        counts = {}
