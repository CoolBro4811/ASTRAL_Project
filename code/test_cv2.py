import cv2
from astropy.io import fits
import sys
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    main(sys.argv[1])
