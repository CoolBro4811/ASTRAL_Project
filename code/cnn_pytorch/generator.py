from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


class StarTrainingGenerator:
    """generate synthetic star images with Gaussian PSF and produce training targets"""

    def __init__(
        self,
        size=256,
        psf_sigma=1.5,
        background=10,
        star_flux=1000,
        read_noise=2.0,
        seed=None,
        target_sigma=2.0,
    ):
        """
        target_sigma: sigma (pixels) for the Gaussian blob in the heatmap target.
        """
        self.size = size
        self.sigma = psf_sigma
        self.background = background
        self.star_flux = star_flux
        self.read_noise = read_noise
        self.target_sigma = target_sigma
        self.rng = np.random.default_rng(seed)

        # avoids recomputing per star
        self.Y, self.X = np.mgrid[0 : self.size, 0 : self.size]

    def gaussian_psf(self, x, y, cx, cy):
        dx = x - cx
        dy = y - cy
        return np.exp(-(dx * dx + dy * dy) / (2 * self.sigma**2))

    def add_star(self, img, cx, cy, mag):
        flux = self.star_flux * 10 ** (-mag / 2.5)

        # vectorized PSF
        psf = np.exp(
            -((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * self.sigma**2)
        )

        img += flux * psf

    def make_heatmap_target(self, labels):
        heatmap = np.zeros((self.size, self.size), dtype=np.float32)
        magmap = np.zeros((self.size, self.size), dtype=np.float32)

        # accumulate weights separately for stable normalization
        mag_weight_sum = np.zeros_like(magmap)

        for star in labels:
            cx, cy, mag = star["x"], star["y"], star["mag"]

            # reuse precomputed grid instead of np.ogrid each loop
            g = np.exp(
                -((self.X - cx) ** 2 + (self.Y - cy) ** 2)
                / (2 * self.target_sigma**2)
            )

            g = g / np.max(g)  # peak = 1

            heatmap = np.maximum(heatmap, g)

            # accumulate magnitude in a weighted way (better learning signal)
            magmap += g * mag
            mag_weight_sum += g

        # normalize AFTER loop (fixes bad per-iteration division)
        mask = mag_weight_sum > 1e-6
        magmap[mask] /= mag_weight_sum[mask]

        return heatmap, magmap

    def generate(self, num_stars, mag_range=(5, 12), margin=10):
        """gen image, labels, heatmap target, and magnitude target"""
        img = np.full((self.size, self.size), self.background, dtype=np.float32)
        labels = []

        for _ in range(num_stars):
            cx = self.rng.uniform(margin, self.size - margin)
            cy = self.rng.uniform(margin, self.size - margin)
            mag = self.rng.uniform(mag_range[0], mag_range[1])

            self.add_star(img, cx, cy, mag)
            labels.append({"x": cx, "y": cy, "mag": mag})

        # noise
        img = self.rng.poisson(np.maximum(img, 0)).astype(np.float32)
        img += self.rng.normal(0, self.read_noise, size=img.shape)
        img = np.clip(img, 0, None)

        heatmap, magmap = self.make_heatmap_target(labels)

        return img, labels, heatmap, magmap

    def save_fits(self, image, labels, filepath):
        """save image and labels as FITS (no heatmap, just for inference)"""
        primary = fits.PrimaryHDU(image.astype(np.float32))
        col_x = fits.Column(
            name="X", format="D", array=[l["x"] for l in labels]
        )
        col_y = fits.Column(
            name="Y", format="D", array=[l["y"] for l in labels]
        )
        col_mag = fits.Column(
            name="MAG", format="D", array=[l["mag"] for l in labels]
        )
        table_hdu = fits.BinTableHDU.from_columns([col_x, col_y, col_mag])
        hdul = fits.HDUList([primary, table_hdu])
        hdul.writeto(filepath, overwrite=True)

    @staticmethod
    def load_fits(filepath):
        with fits.open(filepath) as hdul:
            image = hdul[0].data.astype(np.float32)
            table = hdul[1].data
            labels = [
                {"x": row["X"], "y": row["Y"], "mag": row["MAG"]}
                for row in table
            ]
        return image, labels
