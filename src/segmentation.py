import numpy as np


def label_connected_components(mask):
    """
    Label connected components in boolean mask using 8-connectivity.
    Returns list of (ys, xs) arrays for each component.
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    neigh = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]
                ys = []
                xs = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    ys.append(y)
                    xs.append(x)
                    for dy, dx in neigh:
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                comps.append((np.array(ys, dtype=int), np.array(xs, dtype=int)))
    return comps


def compute_centroid_and_flux(image, ys, xs, subtract_local_bg=False):
    """
    Compute intensity-weighted centroid and total flux.
    If subtract_local_bg True, estimate local background using area around component.
    Returns (x_centroid, y_centroid, flux, peak_value).
    """
    vals = image[ys, xs].astype(float)
    if subtract_local_bg:
        y0, y1 = max(0, ys.min() - 3), min(image.shape[0], ys.max() + 4)
        x0, x1 = max(0, xs.min() - 3), min(image.shape[1], xs.max() + 4)
        local = image[y0:y1, x0:x1]
        mask_local = np.ones_like(local, dtype=bool)
        rel_ys = ys - y0
        rel_xs = xs - x0
        mask_local[rel_ys, rel_xs] = False
        bg = (
            np.median(local[mask_local])
            if np.any(mask_local)
            else np.median(local)
        )
        vals = vals - bg
    total = vals.sum()
    if total <= 0:
        peak_idx = np.argmax(vals)
        return (
            float(xs[peak_idx]),
            float(ys[peak_idx]),
            float(total),
            float(vals.max()),
        )
    x_c = np.sum(xs * vals) / total
    y_c = np.sum(ys * vals) / total
    peak = vals.max()
    return x_c, y_c, total, peak
