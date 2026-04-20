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
        read_noise=2,
        seed=None,
        target_sigma=1.0,
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

    def gaussian_psf(self, x, y, cx, cy):
        dx = x - cx
        dy = y - cy
        return np.exp(-(dx * dx + dy * dy) / (2 * self.sigma**2))

    def add_star(self, img, cx, cy, mag):
        flux = self.star_flux * 10 ** (-mag / 2.5)
        hw = int(5 * self.sigma)
        x0, y0 = int(round(cx)), int(round(cy))
        for dx in range(-hw, hw + 1):
            x = x0 + dx
            if x < 0 or x >= self.size:
                continue
            for dy in range(-hw, hw + 1):
                y = y0 + dy
                if y < 0 or y >= self.size:
                    continue
                psf = self.gaussian_psf(x + 0.5, y + 0.5, cx, cy)
                img[y, x] += flux * psf

    def make_heatmap_target(self, labels):
        """Create a heatmap (Gaussian blobs) and a magnitude map from star labels"""
        heatmap = np.zeros((self.size, self.size), dtype=np.float32)
        magmap = np.zeros((self.size, self.size), dtype=np.float32)
        for star in labels:
            cx, cy, mag = star["x"], star["y"], star["mag"]

            y, x = np.ogrid[: self.size, : self.size]
            g = np.exp(
                -((x - cx) ** 2 + (y - cy) ** 2) / (2 * self.target_sigma**2)
            )
            # normalize
            g = g / np.max(g)
            heatmap = np.maximum(heatmap, g)
            # set magnitude at the peak position
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < self.size and 0 <= iy < self.size:
                magmap[iy, ix] = mag  # magnitude at exact center
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
