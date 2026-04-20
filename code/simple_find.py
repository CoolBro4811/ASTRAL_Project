from astropy.io import fits
import os
import sys
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter
import numpy as np

FWHM = 1.0
SIGMA = FWHM / 3

"""
after calibration:
- images can be modeled as as 
    - I(x,y) = S(x,y) + B(x,y) + N(x,y)
    - where 
        - S(x,y) = signal
        - B(x,y) = residual background
        - N(x,y) = noise

Stars are approximately point sources convolved with telescope PSF:
- S(x,y) = A*PSF(x,y)
    - A = 'amplitude'
    - PSF(x,y) = point spread function at (x,y)

A gaussian is approximately a PSF (kinda), so we can model it as such
- PSF(x,y) ~ exp(frac{x^2 + y^2}{2 * sigma^2})



"""


def gaussian_kernel(size: int, sigma: float):
    """
    create normalized 2d gaussian kernel
    @param size: int
        width/height of kernel in pixels
        i think like ~5*sigma should be fine, right???

    @param sigma: float
        stdev of gaussian in pixels

    @return : 2d np array
        normalized kernel
    """
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel


def matched_filter(image, kernel):
    """
    convolve image with kernel using fft

    O(nlog(n)), better than naive O(n^2) approach

    @param image: 2d array
        calibrated fits image

    @param kernel: 2d array
        gaussian psf kernel

    @return : 2d array
        matched-filtered image
    """
    return fftconvolve(image, kernel, mode="same")


#
#
# def mad_std(image):
#     """
#     mean absolute deviation to find noise level
#
#     @return : float
#         estimated noise stdev
#     """
#
#     med = np.median(image)
#     mad = np.median(np.abs(image - med))
#
#     return mad
#
#
# def snr_map(filtered, sigma):
#     """
#     convert matched filtered image to SNR map
#     """
#     return filtered / sigma
#
#
# def detect_cand(snr, threshold=5):
#     """
#     return bool mask of cand star pixels
#     """
#     return snr > threshold
#
# def find_local_max(image, size=5):
#     """
#     detect local max in image
#     """
#     neighborhood = maximum_filter(image, size=size)
#     return image == neighborhood
#
def detect_stars(snr, threhold=5):
    noise = mad_std(f)
