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
    data *= 255 / np.amax(data)
    data = data.astype(np.uint16)

    print(data)
    print(data.shape)
    rgb = cv2.cvtColor(data, cv2.COLOR_BayerBG2RGB_EA)

    print(rgb)
    visualize(data)
    visualize(rgb)


if __name__ == "__main__":
    main(sys.argv[1])
