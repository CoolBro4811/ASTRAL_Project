from astropy.io import fits
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
from scipy import stats

"""
    files assumed to be in:

    data: fn
    master bias: fn+"bias"
    master dark: fn+"dark"
    master flat: fn+"flat"
"""


def visualize(data):
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.show()


def main(fn: str) -> None:
    # if os.path.exists(fn + "analyzed"):
    #     return

    data = fits.getdata(fn).astype(np.float64)
    visualize(-data)
    if os.path.exists(fn + "bias"):
        bias = fits.getdata(fn + "bias").astype(np.float64)
        data -= bias

    if os.path.exists(fn + "dark"):
        dark = fits.getdata(fn + "dark").astype(np.float64)
        data -= dark

    # subtract bias frame and dark frame
    # assumed to be master bias and master dark, this may be handled later

    if os.path.exists(fn + "flat"):
        flat = fits.getdata(fn + "flat").astype(np.float64)
        m = stats.mode(flat).mode
        flat /= m

        data /= flat
    visualize(data)
    fits.writeto(fn + "analyzed", data)


if __name__ == "__main__":
    main(sys.argv[1])
