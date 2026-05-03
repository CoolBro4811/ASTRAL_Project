import numpy as np
from astropy.io import fits


class SimpleStarGenerator:
    def __init__(
        self,
        size=256,
        psf_sigma=1.5,
        background=10,
        star_flux=1000,
        read_noise=2,
        seed=None,
    ):
        self.size = size
        self.sigma = psf_sigma
        self.background = background
        self.star_flux = star_flux
        self.read_noise = read_noise
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

    def generate(self, num_stars, mag_range=(5, 12), margin=10):
        img = np.full((self.size, self.size), self.background, dtype=np.float32)
        labels = []
        for _ in range(num_stars):
            cx = self.rng.uniform(margin, self.size - margin)
            cy = self.rng.uniform(margin, self.size - margin)
            mag = self.rng.uniform(mag_range[0], mag_range[1])
            self.add_star(img, cx, cy, mag)
            labels.append({"x": cx, "y": cy, "mag": mag})

        img = self.rng.poisson(np.maximum(img, 0)).astype(np.float32)
        img += self.rng.normal(0, self.read_noise, size=img.shape)
        img = np.clip(img, 0, None)
        return img, labels

    def generate_batch(
        self, batch_size, num_stars_range=(5, 30), mag_range=(5, 12)
    ):
        images, all_labels = [], []
        for _ in range(batch_size):
            n = self.rng.integers(num_stars_range[0], num_stars_range[1] + 1)
            img, labs = self.generate(n, mag_range)
            images.append(img)
            all_labels.append(labs)
        return np.stack(images), all_labels

    def save_fits(self, image, labels, filepath):
        """
        Save image and star labels to a FITS file.
        - Primary HDU: image data
        - First extension: binary table with columns X, Y, MAG
        """
        primary = fits.PrimaryHDU(image.astype(np.float32))
        # Create table columns
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
        """Load image and labels from a FITS file saved by save_fits()."""
        with fits.open(filepath) as hdul:
            image = hdul[0].data.astype(np.float32)
            # Read table extension (first extension)
            table = hdul[1].data
            labels = []
            for row in table:
                labels.append({"x": row["X"], "y": row["Y"], "mag": row["MAG"]})
        return image, labels

    def generate_and_save(
        self, num_stars, output_path, mag_range=(5, 12), margin=10
    ):
        """Generate one image and save it directly to a FITS file."""
        img, labels = self.generate(num_stars, mag_range, margin)
        self.save_fits(img, labels, output_path)
        return output_path
