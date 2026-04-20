from astropy.io import fits
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
from scipy import stats

"""
dir: dirname
flat: flat dir name
bias: bias dir name
dark: dark dir name
"""


def visualize(data):
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.show()


def main(dir, fn, bias_n, dark_n, flat_n) -> None:
    # if os.path.exists(fn + "analyzed"):
    #     return

    data = fits.getdata(os.path.join(dir, fn)).astype(np.float64)
    visualize(data)
    if os.path.exists(os.path.join(dir, bias_n)):
        files = os.listdir(os.path.join(dir, bias_n))
        print(files)
        bias = fits.getdata(os.path.join(dir, bias_n, files[0])).astype(
            np.float64
        )
        for file in files[1:]:
            bias += fits.getdata(os.path.join(dir, bias_n, file)).astype(
                np.float64
            )
        bias /= len(files)
        data -= bias

    if os.path.exists(os.path.join(dir, dark_n)):
        files = os.listdir(os.path.join(dir, dark_n))
        print(files)
        dark = fits.getdata(os.path.join(dir, dark_n, files[0])).astype(
            np.float64
        )
        for file in files[1:]:
            dark += fits.getdata(os.path.join(dir, dark_n, file)).astype(
                np.float64
            )
        dark /= len(files)
        data -= dark

    # subtract bias frame and dark frame
    # assumed to be master bias and master dark, this may be handled later

    if os.path.exists(os.path.join(dir, flat_n)):
        files = os.listdir(os.path.join(dir, flat_n))
        print(files)
        flat = fits.getdata(os.path.join(dir, flat_n, files[0])).astype(
            np.float64
        )
        for file in files[1:]:
            flat += fits.getdata(os.path.join(dir, flat_n, file)).astype(
                np.float64
            )
        flat /= len(files)

        m = stats.mode(flat).mode
        flat /= m

        data /= flat
    visualize(data)
    fits.writeto(
        os.path.join(dir, "out", "analyzed.fits"),
        data,
    )


if __name__ == "__main__":
    main(
        os.path.dirname(sys.argv[1]),
        os.path.basename(sys.argv[1]),
        "bias",
        "dark",
        "skyflat",
    )
