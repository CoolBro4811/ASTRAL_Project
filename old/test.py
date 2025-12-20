"""
improved_star_detection.py

Usage:
    python improved_star_detection.py image.fits
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils import CircularAperture, DAOStarFinder, aperture_photometry
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import convolve, gaussian_filter, maximum_filter
from skimage.feature import blob_log


def load_fits(fn):
    with fits.open(fn) as hdul:
        data = hdul[0].data.astype(float)
    # If 3D (e.g., extra dims), try to reduce: take first 2D plane
    if data.ndim > 2:
        data = data.squeeze()
    return data


def robust_background(data, box_size=50, filter_size=3):
    """
    try Background2D if photutils exists
    Returns background, background_rms (noise estimate).
    """
    # clip, median background per box
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(
            data,
            (box_size, box_size),
            filter_size=(filter_size, filter_size),
            bkg_estimator=bkg_estimator,
        )
        return bkg.background, bkg.background_rms
    except Exception:
        pass


def detect_with_photutils(data_sub, fwhm=3.0, threshold_sigma=5.0):
    """
    use DAOStarFinder. data_sub should be background-subtracted image.
    threshold_sigma is in units of background RMS.
    """
    mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)
    threshold = threshold_sigma * std
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    sources = daofind(data_sub - median)
    # sources is an astropy table with xcentroid, ycentroid, flux, sharpness, roundness1/2 etc.
    return sources


def detect_with_blob_log(
    data_sub, min_sigma=1, max_sigma=6, num_sigma=10, threshold=0.02
):
    """
    skimage blob_log returns blobs as (y, x, sigma). threshold is absolute on normalized data.
    Recommend: pass data_sub normalized to [0,1] or adjust threshold relative to max.
    """
    blobs = blob_log(
        data_sub,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )
    # compute approximate radius = sqrt(2) * sigma
    return blobs  # ndarray (n_blobs, 3) with y,x,sigma


def aperture_photometry_snr(
    data, positions, aperture_r=4, bkg_r_in=8, bkg_r_out=12, bkg=None
):
    """
    compute aperture flux and simple SNR.
    positions: list of (x,y) in pixel coords (photutils ordering: x,y)
    returns a list of dicts with x, y, flux, bkg, snr, npix
    """
    results = []
    ny, nx = data.shape
    yy, xx = np.indices(data.shape)
    for x, y in positions:
        # Define masks
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        ap_mask = r2 <= aperture_r**2
        ann_mask = (r2 >= bkg_r_in**2) & (r2 <= bkg_r_out**2)
        flux = np.sum(data[ap_mask])
        npix = np.sum(ap_mask)
        if bkg is None:
            bkg_median = np.median(data[ann_mask]) if np.any(ann_mask) else 0.0
            bkg_rms = np.std(data[ann_mask]) if np.any(ann_mask) else 1.0
        else:
            # use provided local bkg image
            ann_vals = bkg[ann_mask]
            bkg_median = np.median(ann_vals)
            bkg_rms = np.std(ann_vals)
        flux_corr = flux - bkg_median * npix
        # SNR approx: flux / sqrt(flux + npix * sigma_bkg**2), flux in counts
        # Use max(1, flux_corr) to avoid sqrt negative
        sigma_bkg = bkg_rms
        snr = flux_corr / np.sqrt(
            max(flux_corr, 0.0) + npix * sigma_bkg**2 + 1e-12
        )
        results.append(
            {
                "x": x,
                "y": y,
                "flux": flux_corr,
                "npix": npix,
                "bkg": bkg_median,
                "snr": snr,
            }
        )
    return results


def filter_candidates_by_snr(cands, min_snr=5.0, max_fwhm=10.0, min_flux=10.0):
    out = []
    for c in cands:
        if c.get("snr", 0) >= min_snr and c.get("flux", 0) >= min_flux:
            # optionally check fwhm if provided (c['fwhm'])
            out.append(c)
    return out


def run_detection_pipeline(fn, blur_list=None, plot_each=False):
    data = load_fits(fn)
    # handle NaNs or infs
    data = np.nan_to_num(
        data,
        nan=np.median(data[~np.isnan(data)]),
        posinf=np.nanmax(data),
        neginf=np.nanmin(data),
    )
    bkg, bkg_rms = robust_background(data, box_size=64, filter_size=3)
    data_sub = data - bkg
    data_sub = -data_sub

    if blur_list is None:
        blur_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10]

    counts = []
    detections_by_blur = []

    for s in blur_list:
        img = gaussian_filter(data_sub, sigma=s)
        # pptionally scale by original maximum for consistent thresholds (DO NOT normalize to 0-1)
        dets = []
        # Choose fwhm in pixels roughly 2.355*sigma_psf. If image PSF unknown start with fwhm=3
        fwhm = max(2.0, 2.355 * max(s, 1.0))
        try:
            sources = detect_with_photutils(img, fwhm=fwhm, threshold_sigma=5.0)
            if sources is not None:
                # convert sources table to positions (xcentroid, ycentroid)
                positions = [
                    (row["xcentroid"], row["ycentroid"]) for row in sources
                ]
                phot = aperture_photometry_snr(
                    img, positions, aperture_r=max(3, int(round(fwhm)))
                )
                dets = phot
        except Exception as e:
            print("photutils detection failed:", e)
            dets = []
            # filter by SNR and flux
        dets_f = filter_candidates_by_snr(dets, min_snr=5.0, min_flux=20.0)
        counts.append(len(dets_f))
        detections_by_blur.append((s, dets_f))
        print(f"blur sigma={s}: detected {len(dets_f)} stars")

        if plot_each:
            plt.figure(figsize=(8, 6))
            plt.imshow(
                -img,
                origin="lower",
                cmap="gray",
                vmin=np.percentile(img, 5),
                vmax=np.percentile(img, 99),
            )
            xs = [d["x"] for d in dets_f]
            ys = [d["y"] for d in dets_f]
            plt.scatter(xs, ys, s=40, edgecolor="red", facecolor="none")
            plt.title(f"Blur {s} detections: {len(dets_f)}")
            plt.show()

    # summary plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(data, 5),
        vmax=np.percentile(data, 99),
    )
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(blur_list, counts, marker="o")
    plt.xlabel("Gaussian blur sigma")
    plt.ylabel("Detected stars")
    plt.title("Detections vs blur")

    plt.tight_layout()
    plt.show()

    return detections_by_blur, counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python improved_star_detection.py image.fits")
        sys.exit(1)
    fn = sys.argv[1]
    if not os.path.exists(fn):
        print("File not found:", fn)
        sys.exit(1)
    run_detection_pipeline(fn, blur_list=[0.5, 1, 2, 3, 5], plot_each=False)
