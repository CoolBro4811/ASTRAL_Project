import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder


def get_sources(data, threshold_sigma, fwhm):
    _, med, std = sigma_clipped_stats(data)
    threshold = threshold_sigma * std
    sources = DAOStarFinder(threshold=threshold, fwhm=fwhm)(data - med)
    count = len(sources) if sources is not None else 0
    return sources, count


def detect(filepath: str):
    print(f"detecting data for {filepath}")
    data = fits.getdata(filepath)
    thresholds = [4, 5, 6, 7]  # edit for better data?
    fwhms = [2.5, 3.0, 3.5, 4.5]
    best_count = -1
    best_params = None
    best_sources = None
    for t in thresholds:
        for f in fwhms:
            sources, count = get_sources(data, t, f)
            if count > best_count:
                best_count = count
                best_params = (t, f)
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
