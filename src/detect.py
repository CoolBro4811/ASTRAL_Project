import numpy as np

from .background import blockwise_median_background
from .fitting import SCIPY_AVAILABLE, fit_gaussian_around
from .segmentation import compute_centroid_and_flux, label_connected_components
from .utils import fft_convolve, mad_std, make_gaussian_kernel


def detect_stars(
    image,
    k=5.0,
    psf_sigma=1.5,
    bg_block=64,
    matched_filter=True,
    min_area=3,
    fit_gaussian=True,
    invert=False,
):
    """
    Full detection pipeline:
     - background estimation (blockwise median)
     - background subtraction
     - MAD-based noise estimate
     - optional matched filtering with a Gaussian of width psf_sigma
     - threshold at median + k*sigma
     - connected components and centroid/flux calculation
     - optional Gaussian fit (if SciPy is available)
    Returns (detections_list, aux_dict)
    """
    img = np.array(image, dtype=float)
    if invert:
        img = -img
    bg = blockwise_median_background(img, block_size=bg_block)
    img_sub = img - bg
    med = np.median(img_sub)
    sigma = mad_std(img_sub)
    if matched_filter and psf_sigma is not None:
        ksize = int(max(3, int(8 * psf_sigma) // 2 * 2 + 1))
        gk = make_gaussian_kernel(ksize, psf_sigma)
        img_filt = fft_convolve(img_sub, gk)
    else:
        img_filt = img_sub.copy()
    thr = med + k * sigma
    mask = img_filt > thr
    comps = label_connected_components(mask)
    detections = []
    for ys, xs in comps:
        if ys.size < min_area:
            continue
        x_c, y_c, flux, peak = compute_centroid_and_flux(
            img_sub, ys, xs, subtract_local_bg=True
        )
        det = {
            "x": x_c,
            "y": y_c,
            "flux": flux,
            "peak": peak,
            "area": int(ys.size),
        }
        if fit_gaussian and SCIPY_AVAILABLE:
            fit = fit_gaussian_around(img, y_c, x_c, radius=6)
            if fit is not None:
                det["fit"] = fit
        detections.append(det)
    detections.sort(key=lambda d: d["peak"], reverse=True)
    aux = {
        "background": bg,
        "img_sub": img_sub,
        "img_filt": img_filt,
        "threshold": thr,
    }
    return detections, aux
