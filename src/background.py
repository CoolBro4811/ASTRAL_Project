import numpy as np

from .utils import fft_convolve, make_gaussian_kernel


def blockwise_median_background(image, block_size=64):
    """
    Estimate background by medians on non-overlapping blocks and smooth.
    Returns background image same shape as input.
    """
    H, W = image.shape
    bh = int(np.ceil(H / block_size))
    bw = int(np.ceil(W / block_size))
    pad_h = bh * block_size - H
    pad_w = bw * block_size - W
    img_pad = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    med = np.zeros((bh, bw))
    for i in range(bh):
        for j in range(bw):
            block = img_pad[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            med[i, j] = np.median(block)
    up = np.repeat(np.repeat(med, block_size, axis=0), block_size, axis=1)
    up = up[: img_pad.shape[0], : img_pad.shape[1]]
    small_k = make_gaussian_kernel(9, sigma=1.5)
    up_smooth = fft_convolve(up, small_k)
    bg = up_smooth[:H, :W]
    return bg
