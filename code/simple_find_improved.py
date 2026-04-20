"""
Star detection in astronomical images using a matched filter with a Gaussian PSF.

The algorithm:
1. Read the FITS image.
2. Estimate and subtract the background (median from sigma‑clipped statistics).
3. Estimate the noise standard deviation from the background‑subtracted image.
4. Build a normalized Gaussian kernel from the given FWHM.
5. Apply a matched filter (FFT convolution) to enhance point sources.
6. Compute the SNR map: filtered image / (noise × sqrt(∑ kernel²)).
7. Threshold at a user‑defined SNR level.
8. Find local maxima within the thresholded region.
9. Optionally refine the positions by 2‑D centroiding.

Visualization:
- Use `visualize_detections` to plot the original image with detected stars marked,
  and optionally show the SNR map side‑by‑side.

Dependencies: numpy, scipy, astropy, matplotlib (for visualization)
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter, label
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

__all__ = [
    "gaussian_kernel",
    "matched_filter",
    "estimate_background_noise",
    "detect_stars",
    "visualize_detections",
]


def gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    """
    Create a normalized 2‑D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian (in pixels).
    size : int, optional
        Side length of the square kernel. If None, it is set to
        2 * ceil(3 * sigma) + 1, which contains > 99.7% of the mass.

    Returns
    -------
    kernel : 2‑D numpy array
        Normalized kernel (sum = 1).
    """
    if size is None:
        size = (
            2 * int(np.ceil(3 * sigma)) + 1
        )  # odd, covers most of the Gaussian
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def matched_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve the image with the kernel using FFT (fast).

    Parameters
    ----------
    image : 2‑D numpy array
        Input image (background‑subtracted).
    kernel : 2‑D numpy array
        Normalized kernel.

    Returns
    -------
    filtered : 2‑D numpy array
        Matched‑filtered image (same size as input).
    """
    return fftconvolve(image, kernel, mode="same")


def estimate_background_noise(
    image: np.ndarray, sigma_clip: float = 3.0
) -> tuple:
    """
    Estimate the background level and the noise standard deviation.

    Uses sigma‑clipping to exclude pixels likely belonging to stars.

    Parameters
    ----------
    image : 2‑D numpy array
        The original image.
    sigma_clip : float, optional
        Number of standard deviations for sigma‑clipping.

    Returns
    -------
    background : float
        Estimated background level (median of the clipped distribution).
    noise : float
        Estimated standard deviation of the noise.
    """
    _, background, noise = sigma_clipped_stats(
        image, sigma=sigma_clip, maxiters=5
    )
    return background, noise


def detect_stars(
    filename: str,
    fwhm: float = 1.0,
    snr_threshold: float = 5.0,
    min_distance: float = None,
    refine_positions: bool = True,
    edge_margin: int = None,
    return_maps: bool = False,
) -> list:
    """
    Detect stars in a FITS image using a matched filter.

    Parameters
    ----------
    filename : str
        Path to the FITS file.
    fwhm : float, optional
        Full width at half maximum of the PSF (in pixels). Default is 1.0.
    snr_threshold : float, optional
        Signal‑to‑noise ratio threshold for detection. Default is 5.0.
    min_distance : float, optional
        Minimum distance (in pixels) between two detected stars.
        If None, it is set to the FWHM.
    refine_positions : bool, optional
        If True, refine the positions by 2‑D centroiding in a small window.
    edge_margin : int, optional
        Number of pixels from the edges to ignore. If None, it is set to
        2 * ceil(3 * sigma) where sigma = fwhm / 2.355.
    return_maps : bool, optional
        If True, also return the SNR map and the filtered image.

    Returns
    -------
    stars : list of tuples
        Each tuple is (x, y, snr) where (x, y) are the centroid coordinates
        (0‑based array indices) and snr is the peak SNR.
    snr_map : 2‑D numpy array (if return_maps=True)
        The SNR map.
    filtered : 2‑D numpy array (if return_maps=True)
        The matched‑filtered image (background‑subtracted).
    """
    # 1. Read the FITS image
    with fits.open(filename) as hdul:
        image = hdul[0].data.astype(float)
        # If the image is 3‑D (e.g., some FITS have a dummy axis), take the first plane
        if image.ndim == 3:
            image = image[0]
        elif image.ndim != 2:
            raise ValueError(f"Expected 2‑D image, got {image.ndim}‑D")

    # 2. Estimate and subtract background
    background, noise_original = estimate_background_noise(image)
    image_sub = image - background

    # 3. Create the Gaussian kernel
    sigma = fwhm / 2.355  # correct relation for Gaussian FWHM
    kernel = gaussian_kernel(sigma)

    # 4. Apply matched filter
    filtered = matched_filter(image_sub, kernel)

    # 5. Compute the noise in the filtered image
    kernel_sq_sum = np.sum(kernel**2)
    noise_filtered = noise_original * np.sqrt(kernel_sq_sum)

    # 6. Compute SNR map
    snr_map = filtered / noise_filtered

    # 7. Define the edge margin
    if edge_margin is None:
        edge_margin = 2 * int(np.ceil(3 * sigma))
    ny, nx = snr_map.shape
    edge_mask = np.ones_like(snr_map, dtype=bool)
    edge_mask[:edge_margin, :] = False
    edge_mask[-edge_margin:, :] = False
    edge_mask[:, :edge_margin] = False
    edge_mask[:, -edge_margin:] = False

    # 8. Threshold the SNR map
    threshold_mask = (snr_map > snr_threshold) & edge_mask

    # 9. Find local maxima
    if min_distance is None:
        min_distance = fwhm
    neighborhood_size = int(np.ceil(min_distance))
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    max_filtered = maximum_filter(
        snr_map, size=neighborhood_size, mode="constant", cval=-np.inf
    )
    local_max = (snr_map == max_filtered) & threshold_mask

    # 10. Label the peaks
    labeled, num_peaks = label(local_max)

    stars = []
    for i in range(1, num_peaks + 1):
        y_peak, x_peak = np.where(labeled == i)
        y_peak, x_peak = y_peak[0], x_peak[0]

        if refine_positions:
            half_window = int(np.ceil(3 * sigma))
            y_min = max(y_peak - half_window, 0)
            y_max = min(y_peak + half_window + 1, ny)
            x_min = max(x_peak - half_window, 0)
            x_max = min(x_peak + half_window + 1, nx)
            window = image_sub[y_min:y_max, x_min:x_max]
            total_flux = np.sum(window)
            if total_flux > 0:
                y_grid, x_grid = np.indices(window.shape)
                x_cent = np.sum(x_grid * window) / total_flux + x_min
                y_cent = np.sum(y_grid * window) / total_flux + y_min
            else:
                x_cent, y_cent = x_peak, y_peak
        else:
            x_cent, y_cent = x_peak, y_peak

        snr_peak = snr_map[y_peak, x_peak]
        stars.append((x_cent, y_cent, snr_peak))

    if return_maps:
        return stars, snr_map, filtered
    else:
        return stars


def visualize_detections(
    filename: str,
    stars: list,
    snr_map: np.ndarray = None,
    figsize: tuple = (12, 5),
    cmap_image: str = "gray",
    cmap_snr: str = "viridis",
    marker_style: dict = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Visualize the detected stars on the original image and optionally on the SNR map.

    Parameters
    ----------
    filename : str
        Path to the FITS file (used to load the image).
    stars : list of tuples
        List of (x, y, snr) detected stars.
    snr_map : 2‑D numpy array, optional
        The SNR map (if available) to display in a second subplot.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap_image : str, optional
        Colormap for the original image.
    cmap_snr : str, optional
        Colormap for the SNR map.
    marker_style : dict, optional
        Dictionary of keyword arguments for the scatter plot markers.
        Default: {'s': 80, 'facecolors': 'none', 'edgecolors': 'red', 'linewidths': 1.5}
    show : bool, optional
        If True, call plt.show().
    save_path : str, optional
        If provided, save the figure to this path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for visualization. Install it with 'pip install matplotlib'."
        )

    # Load image
    with fits.open(filename) as hdul:
        image = hdul[0].data.astype(float)
        if image.ndim == 3:
            image = image[0]
        elif image.ndim != 2:
            raise ValueError(f"Expected 2‑D image, got {image.ndim}‑D")

    # Prepare marker style
    if marker_style is None:
        marker_style = {
            "s": 80,
            "facecolors": "none",
            "edgecolors": "red",
            "linewidths": 1.5,
        }

    # Extract coordinates and SNRs
    xs = [s[0] for s in stars]
    ys = [s[1] for s in stars]
    snrs = [s[2] for s in stars]

    # Determine number of subplots
    if snr_map is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # Show SNR map
        im2 = ax2.imshow(
            snr_map, origin="lower", cmap=cmap_snr, interpolation="nearest"
        )
        ax2.set_title("SNR Map")
        plt.colorbar(im2, ax=ax2, label="SNR")
        ax2.scatter(xs, ys, **marker_style)
        ax2.set_xlim(0, image.shape[1])
        ax2.set_ylim(0, image.shape[0])
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
        ax2 = None

    # Show original image (with scaling for display)
    vmin, vmax = np.percentile(image, (1, 99))  # stretch for visibility
    im1 = ax1.imshow(
        image,
        origin="lower",
        cmap=cmap_image,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax1.set_title("Original Image with Detected Stars")
    plt.colorbar(im1, ax=ax1, label="Flux")

    # Overlay star positions
    ax1.scatter(xs, ys, **marker_style)
    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(0, image.shape[0])

    # Annotate with SNR (optional)
    for x, y, snr in stars:
        ax1.text(
            x,
            y,
            f"{snr:.1f}",
            color="cyan",
            fontsize=8,
            ha="center",
            va="bottom",
            weight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python star_detection.py <fits_file> [fwhm] [snr_threshold] [--show]"
        )
        sys.exit(1)

    fname = sys.argv[1]
    fwhm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    snr_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    # Detect stars and also get the SNR map for visualization
    stars, snr_map, _ = detect_stars(
        fname, fwhm=fwhm, snr_threshold=snr_thr, return_maps=True
    )
    print(f"Found {len(stars)} stars")
    for i, (x, y, snr) in enumerate(stars):
        print(f"Star {i+1}: x={x:.2f}, y={y:.2f}, SNR={snr:.2f}")

    # Visualize
    visualize_detections(fname, stars, snr_map=snr_map)
