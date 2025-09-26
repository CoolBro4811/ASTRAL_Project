import numpy as np
from numpy.fft import fft2, ifft2


def make_gaussian_kernel(size, sigma):
    """Return square Gaussian kernel (size odd) normalized to sum=1."""
    assert size % 2 == 1, "size should be odd"
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def fft_convolve(image, kernel):
    """Convolve image with kernel using FFT; crop to original image size."""
    s1 = np.array(image.shape)
    s2 = np.array(kernel.shape)
    shape = s1 + s2 - 1
    fimage = fft2(image, shape)
    fkernel = fft2(kernel, shape)
    conv = np.real(ifft2(fimage * fkernel))
    start = (s2 - 1) // 2
    end = start + s1
    ys = slice(start[0], end[0])
    xs = slice(start[1], end[1])
    return conv[ys, xs]


def mad_std(data):
    """Estimate standard deviation using the median absolute deviation (MAD)."""
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return 1.4826 * mad
