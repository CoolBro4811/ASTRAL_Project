import numpy as np

try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def twoD_gaussian(coords, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian model. coords is (y_flat, x_flat)."""
    y, x = coords
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    return offset + amp * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )


def fit_gaussian_around(image, yc, xc, radius=6):
    """
    Fit 2D Gaussian to sub-image centered near (yc, xc).
    Returns dict with keys 'amp','x','y','sx','sy','theta','offset' or None if unavailable/failed.
    """
    if not SCIPY_AVAILABLE:
        return None
    H, W = image.shape
    y0 = int(np.clip(np.round(yc) - radius, 0, H - 1))
    y1 = int(np.clip(np.round(yc) + radius + 1, 0, H))
    x0 = int(np.clip(np.round(xc) - radius, 0, W - 1))
    x1 = int(np.clip(np.round(xc) + radius + 1, 0, W))
    sub = image[y0:y1, x0:x1].astype(float)
    ys, xs = np.mgrid[0 : sub.shape[0], 0 : sub.shape[1]]
    coords = (ys.ravel(), xs.ravel())
    amp0 = sub.max() - np.median(sub)
    x0_init = (xs * sub).sum() / sub.sum()
    y0_init = (ys * sub).sum() / sub.sum()
    p0 = [amp0, x0_init, y0_init, 1.5, 1.5, 0.0, np.median(sub)]
    try:
        popt, _ = curve_fit(
            lambda flat_coords, amp, x0i, y0i, sx, sy, th, off: twoD_gaussian(
                (coords[0], coords[1]), amp, x0i, y0i, sx, sy, th, off
            ),
            np.concatenate([coords[0], coords[1]]),
            sub.ravel(),
            p0=p0,
            maxfev=20000,
        )
        amp, x0f, y0f, sx, sy, th, off = popt
        return {
            "amp": amp,
            "x": x0 + x0f,
            "y": y0 + y0f,
            "sx": sx,
            "sy": sy,
            "theta": th,
            "offset": off,
        }
    except Exception:
        return None
