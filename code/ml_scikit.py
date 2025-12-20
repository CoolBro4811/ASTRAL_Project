import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def estimate_background(data, mask=None, sigma=3.0, iters=5):
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    return sigma_clipped_stats(data, sigma=sigma, maxiters=iters, mask=mask)


def detect_with_photutils(data, median, std, fwhm=3.0, threshold_sigma=5.0):
    """
    DAOStarFinder to detect_sources
    ret: positions array (y,x), photutils table (or None)
    """
    threshold = median + threshold_sigma * std
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    sources = daofind(data - median)
    if sources is None or len(sources) == 0:
        print("warning: DAOStarFinder found 0 sources")
        return np.empty((0, 2)), sources
    # photutils returns xcentroid, ycentroid: make (y,x) pairs
    positions = np.vstack([sources["ycentroid"], sources["xcentroid"]]).T
    return positions, sources


def aperture_photometry_basic(data, positions, r=3):
    """
    basic circular aperture photometry, sum of pixels inside radius r around (y,x)
    ret: flux + peak values
    """
    fluxes = []
    peaks = []
    ny, nx = data.shape

    for y, x in positions:
        yi = int(round(y))
        xi = int(round(x))
        y0 = max(0, yi - r)
        y1 = min(ny, yi + r + 1)
        x0 = max(0, xi - r)
        x1 = min(nx, xi + r + 1)

        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]

        dist2 = (yy - yi) ** 2 + (xx - xi) ** 2
        mask = dist2 <= r * r
        sub = data[y0:y1, x0:x1]
        if sub.size == 0:
            flux = 0.0
        else:
            flux = sub[mask].sum()
        # guard bounds for peak
        peak = float(data[yi, xi]) if 0 <= yi < ny and 0 <= xi < nx else 0.0
        fluxes.append(float(flux))
        peaks.append(float(peak))
    return np.array(fluxes), np.array(peaks)


def compute_second_moments(data, positions, r=4):
    """
    find simple second moments inside a window to estimate size/elongation.
    return array shape (N,3): sig_x, sig_y, ecc
    pixels > 0 only, (assumed to be correct)
    """
    feats = []
    ny, nx = data.shape
    for y, x in positions:
        yi = int(round(y))
        xi = int(round(x))
        y0 = max(0, yi - r)
        y1 = min(ny, yi + r + 1)
        x0 = max(0, xi - r)
        x1 = min(nx, xi + r + 1)

        sub = data[y0:y1, x0:x1].astype(float)
        if sub.size == 0:
            feats.append((0.0, 0.0, 0.0))
            continue

        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]

        mask = sub > 0  # consider positive pixels
        if not np.any(mask):
            feats.append((0.0, 0.0, 0.0))
            continue

        weights = sub
        ycoords = yy - yi
        xcoords = xx - xi
        total = weights.sum() + 1e-12

        mxx = (weights * (xcoords**2)).sum() / total
        myy = (weights * (ycoords**2)).sum() / total
        # mxy not used directly but computed for completeness
        mxy = (weights * (xcoords * ycoords)).sum() / total

        sig_x = float(np.sqrt(mxx))
        sig_y = float(np.sqrt(myy))
        # protect division by zero errors ive been having
        denom = max(sig_x, sig_y) ** 2 + 1e-12
        ecc = float(np.sqrt(max(0.0, 1.0 - min(sig_x, sig_y) ** 2 / denom)))
        feats.append((sig_x, sig_y, ecc))
    return np.array(feats)


def merge_positions(positions, fluxes, method="dbscan", merge_eps=2.5):
    """
    merge nearby detections into one representative per spatial cluster.
    - positions: (N,2) array of (y,x)
    - fluxes: array length N
    - method: 'dbscan', 'single' (single-link agglomerative), 'none'
    - merge_eps: distance threshold in pixels
    returns:
      final_positions (M,2), mapping indices (list of lists of original indices)
    """
    if positions.shape[0] == 0:
        return positions, []

    if method == "none":
        mapping = [[i] for i in range(len(positions))]
        return positions.copy(), mapping

    if method == "dbscan":
        # DBSCAN on spatial coordinates. Use min_samples=1 so isolated points survive.
        db = DBSCAN(eps=merge_eps, min_samples=1).fit(positions)
        labels = db.labels_
        unique = np.unique(labels)
        mapping = []
        reps = []
        for lbl in unique:
            members = np.where(labels == lbl)[0]
            # choose representative: highest flux if available, else first
            rep = (
                members[np.argmax(fluxes[members])]
                if len(members) > 0
                else members[0]
            )
            reps.append(positions[rep])
            mapping.append(list(members))
        final_positions = np.vstack(reps)
        return final_positions, mapping

    if method == "single":
        # Agglomerative single-link clustering with distance cutoff.
        # sklearn's AgglomerativeClustering supports distance_threshold from v0.21+
        # We'll use pairwise distances and single-link thresholding using scipy if available,
        # fallback to sklearn AgglomerativeClustering (if available).
        try:
            # compute condensed distance matrix and apply single-link clustering by threshold
            dists = squareform(pdist(positions))
            # build adjacency where distance <= merge_eps
            # form connected components of that adjacency (single-link)
            vis = np.zeros(len(positions), dtype=bool)
            mapping = []
            for i in range(len(positions)):
                if vis[i]:
                    continue
                # BFS to find all connected within threshold
                stack = [i]
                comp = []
                while stack:
                    u = stack.pop()
                    if vis[u]:
                        continue
                    vis[u] = True
                    comp.append(u)
                    neighbors = np.where(dists[u] <= merge_eps)[0]
                    for nb in neighbors:
                        if not vis[nb]:
                            stack.append(int(nb))
                mapping.append(sorted(comp))
            # pick representative by flux
            reps = []
            for members in mapping:
                rep = members[np.argmax(fluxes[members])]
                reps.append(positions[rep])
            final_positions = (
                np.vstack(reps) if len(reps) > 0 else np.empty((0, 2))
            )
            return final_positions, mapping
        except Exception:
            # fallback: no scipy/squareform/pdist - just do DBSCAN
            return merge_positions(
                positions, fluxes, method="dbscan", merge_eps=merge_eps
            )

    raise ValueError(f"Unknown merge method: {method}")


# --- main pipeline --- #
def star_count_pipeline(
    fits_path,
    outdir=".",
    plot=False,
    threshold_sigma=4.0,
    fwhm_guess=3.0,
    aperture_r=3,
    footprint_size=5,
    merge_eps=2.5,
    merge_method="dbscan",
):
    """
    run pipeline on a FITS file.
    returns results dict with detections, features, and final star count.
    """
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    hdul = fits.open(fits_path)
    data = hdul[0].data.astype(float)
    hdul.close()

    if data.ndim > 2:
        data = data[0]
        print("warning: >2 dims, choosing first 2D slice")

    mean, median, std = estimate_background(data)
    print(
        f"Background estimate: mean={mean:.3f}, median={median:.3f}, std={std:.3f}"
    )

    thresh = median + threshold_sigma * std
    print(
        f"Detection threshold set to median + {threshold_sigma}*std = {thresh:.3f}"
    )

    positions, sources = detect_with_photutils(
        data, median, std, fwhm=fwhm_guess, threshold_sigma=threshold_sigma
    )

    # fallback if photutils finds nothing (positions may be empty)
    if positions is None or positions.shape[0] == 0:
        print("No positions detected. Exiting.")
        results = {
            "fits_path": fits_path,
            "positions": positions,
            "count": 0,
            "sources_table": sources,
        }
        return results

    fluxes, peaks_vals = aperture_photometry_basic(
        data, positions, r=aperture_r
    )
    moments = compute_second_moments(data, positions, r=4)

    # feature matrix for GMM: log(flux), log(peak), sig_x, sig_y, ecc
    X = np.column_stack(
        [
            np.log10(np.maximum(fluxes, 1e-8)),
            np.log10(np.maximum(peaks_vals, 1e-8)),
            moments[:, 0],
            moments[:, 1],
            moments[:, 2],
        ]
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # GaussianMixture to separate point-like from extended/noise
    try:
        gmm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=67
        )
        labels = gmm.fit_predict(Xs)
        comp0_median_flux = (
            np.median(fluxes[labels == 0]) if np.any(labels == 0) else 0
        )
        comp1_median_flux = (
            np.median(fluxes[labels == 1]) if np.any(labels == 1) else 0
        )
        star_comp = 0 if comp0_median_flux > comp1_median_flux else 1
        is_star = labels == star_comp
        print(
            f"GMM separated into 2 components; star-like component = {star_comp}."
        )
    except Exception as e:
        print("GMM failed; falling back to simple flux > median rule.", e)
        is_star = fluxes > np.median(fluxes)

    # Merge nearby detections (positions-only). This will not drop isolated points (unless method='none').
    final_positions, mapping = merge_positions(
        positions, fluxes, method=merge_method, merge_eps=merge_eps
    )

    # Build final is_star flags corresponding to final_positions: a cluster is star-like if any member was star-like
    final_is_star = []
    for members in mapping:
        final_is_star.append(bool(np.any(is_star[members])))
    final_is_star = np.array(final_is_star, dtype=bool)
    final_star_count = int(final_is_star.sum())

    print(
        f"Merged {len(positions)} detections -> {len(final_positions)} unique detections after '{merge_method}' merging."
    )
    print(f"Final star-like object count = {final_star_count}")

    # Save CSV summary
    df = pd.DataFrame(
        {
            "y": positions[:, 0],
            "x": positions[:, 1],
            "flux": fluxes,
            "peak": peaks_vals,
            "sigx": moments[:, 0],
            "sigy": moments[:, 1],
            "ecc": moments[:, 2],
            "is_star_initial": is_star,
        }
    )
    csv_path = os.path.join(
        outdir, f"{os.path.basename(fits_path)}_detections_summary.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Detections CSV -> {csv_path}")

    # Diagnostic plot
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=median - std,
            vmax=median + 6 * std,
        )
        if positions.shape[0] > 0:
            ax.scatter(
                positions[:, 1],
                positions[:, 0],
                marker="x",
                color="red",
                s=30,
                label="initial positions",
            )
        if final_positions.shape[0] > 0:
            starpos = final_positions[final_is_star]
            ax.scatter(
                starpos[:, 1],
                starpos[:, 0],
                facecolors="none",
                edgecolors="yellow",
                s=80,
                linewidths=1.2,
                label="final star-like",
            )
        ax.set_title(
            f"Detections: initial {len(positions)}, final star-like {final_star_count}"
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        plot_path = os.path.join(
            outdir, f"{os.path.basename(fits_path)}_diagnostic.png"
        )
        plt.savefig(plot_path, dpi=200)
        plt.show()
        print(f"Diagnostic plot -> {plot_path}")

    results = {
        "fits_path": fits_path,
        "positions": positions,
        "fluxes": fluxes,
        "peaks_vals": peaks_vals,
        "moments": moments,
        "X": X,
        "Xs": Xs,
        "is_star_initial": is_star,
        "final_positions": final_positions,
        "final_is_star": final_is_star,
        "mapping": mapping,
        "count": final_star_count,
        "csv": csv_path,
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Star counting pipeline with scikit-learn (improved merging)."
    )
    parser.add_argument("fits", help="Input FITS path")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument(
        "--plot", action="store_true", help="Show and save diagnostic plot"
    )
    parser.add_argument(
        "--thresh_sigma",
        type=float,
        default=4.0,
        help="Threshold in sigma above median",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=3.0,
        help="Approx FWHM (pixels) for DAO detection",
    )
    parser.add_argument(
        "--aperture",
        type=int,
        default=3,
        help="Aperture radius (pixels) for photometry",
    )
    parser.add_argument(
        "--merge_eps",
        type=float,
        default=2.5,
        help="Merge radius (pixels) for spatial merging",
    )
    parser.add_argument(
        "--merge_method",
        choices=["dbscan", "single", "none"],
        default="dbscan",
        help="Merging algorithm for close detections",
    )
    args = parser.parse_args()

    res = star_count_pipeline(
        args.fits,
        outdir=args.outdir,
        plot=args.plot,
        threshold_sigma=args.thresh_sigma,
        fwhm_guess=args.fwhm,
        aperture_r=args.aperture,
        merge_eps=args.merge_eps,
        merge_method=args.merge_method,
    )
    print(
        "Done. Results summary:",
        {k: res[k] for k in ("fits_path", "count", "csv")},
    )


if __name__ == "__main__":
    main()
