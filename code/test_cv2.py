import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def visualize(data):
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.show()


def main(fn):
    data = fits.getdata(fn).astype(np.float64)
    data *= 255 / np.amax(data)  # normalize for regular rgb space
    data = data.astype(np.uint16)  # cv2 cvtcolor requires uint8 or uint16
    rgb = cv2.cvtColor(
        data, cv2.COLOR_BayerBG2RGB_EA
    )  # BG -> RGB, with edge aware

    print(rgb)
    visualize(data)
    visualize(rgb)
    # visualize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))


def get_debayered(data):
    data *= 255 / np.amax(data)  # normalize for regular rgb space
    data = data.astype(np.uint16)  # cv2 cvtcolor requires uint8 or uint16
    rgb = cv2.cvtColor(
        data, cv2.COLOR_BayerBG2RGB_EA
    )  # BG -> RGB, with edge aware

    return rgb


def get_debayered_white(data):
    data *= 255 / np.amax(data)  # normalize for regular rgb space
    data = data.astype(np.uint16)  # cv2 cvtcolor requires uint8 or uint16
    rgb = cv2.cvtColor(
        data, cv2.COLOR_BayerBG2RGB_EA
    )  # BG -> RGB, with edge aware
    white = np.sum(rgb * [1 / 3, 1 / 3, 1 / 3], axis=-1)
    return white

    return rgb


if __name__ == "__main__":
    main(sys.argv[1])
