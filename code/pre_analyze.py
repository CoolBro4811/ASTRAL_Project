from astropy.io import fits
import numpy as np
import sys
from scipy import stats


def main(fn: str):
    if os.path.exists(fn + "analyzed"):
        return

    data = fits.open(fn).data[0].astype(float64)

    bias = fits.open(fn + "bias").data[0].astype(float64)

    dark = fits.open(fn + "dark").data[0].astype(float64)

    data -= bias + dark

    flat = fits.open(fn + "flat").data[0].astype(float64)

    m = stats.mode(flat)

    data /= m

    data.writeto(fn + "analyzed")


if __name__ == "__main__":
    main(sys.argv[1])
