import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder

fits_file = sys.argv[1]
saturation_limit = 50000  # look at notes?
k = 4
ZP = 25.0  # needs some more testing - just chooisng random value
# this wil convert counts to true mag - base off of a measurement with known magnitudes
read_noise = 5.0  # we can find this later?
gain = 1.0  # pixel counts - electrons (idk yet)

data = fits.getdata(fits_file).astype(float)
data[data > saturation_limit] = np.nan

# background and noise
bkg_estimator = MedianBackground()
bkg = Background2D(
    data, box_size=64, filter_size=3, bkg_estimator=bkg_estimator
)
bkg_mean = bkg.background_median
bkg_sigma = bkg.background_rms_median

data_sub = data - bkg.background

# detect stars
daofind = DAOStarFinder(fwhm=3.0, threshold=k * bkg_sigma)
sources = daofind(data_sub)
if sources is None or len(sources) == 0:
    raise RuntimeError("No stars detected!")

x, y = sources["xcentroid"], sources["ycentroid"]

# aperture photometry
aperture = CircularAperture(np.column_stack((x, y)), r=4.0)
phot_table = aperture_photometry(data_sub, aperture)

# calculate instrumental magnitudes
flux = phot_table["aperture_sum"] * gain
mag = -2.5 * np.log10(flux) + ZP

# estimate SNR
n_pix = np.pi * (4**2)
B = bkg_mean * gain
SNR = flux / np.sqrt(flux + n_pix * (B + read_noise**2))

# combine results
df = pd.DataFrame({"x": x, "y": y, "flux": flux, "mag": mag, "SNR": SNR})
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# limiting mag (SNR >= 5???)
limiting_mag = df.loc[df["SNR"] >= 5, "mag"].max()
print(f"Limiting magnitude (SNR≥5): {limiting_mag:.2f}")

# save and plot
df.to_csv("star_counts.csv", index=False)
plt.scatter(df["mag"], df["SNR"], s=10)
plt.axhline(5, color="r", linestyle="--", label="SNR=5")
plt.gca().invert_xaxis()
plt.xlabel("Instrumental Magnitude")
plt.ylabel("SNR")
plt.legend()
plt.tight_layout()
plt.savefig("snr_vs_mag.png", dpi=150)
plt.show()
