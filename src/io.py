import numpy as np
from astropy.io import fits


def load_fits_image(filename, ext=0):
    """
    Load a FITS image and return it as a 2D numpy array (float).
    - filename: path to FITS file
    - ext: HDU extension index (default 0 = primary)
    """
    with fits.open(filename, memmap=False) as hdul:
        data = hdul[ext].data
        if data is None:
            raise ValueError(f"No data in HDU {ext} of {filename}")
        data = np.asarray(data, dtype=float)
    if data.ndim > 2:
        # Choose first slice if 3+D (common for data cubes)
        data = data[0]
    return data


def save_catalog_csv(detections, filename, fields=None):
    """
    Save detections (list of dicts) to CSV.
    fields: list of keys (columns) to include; default: ['x','y','flux','peak','area']
    """
    import csv

    if fields is None:
        fields = ["x", "y", "flux", "peak", "area"]

    with open(filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for d in detections:
            row = []
            for f in fields:
                v = d.get(f, "")
                # If v is a dict (fit), skip unless explicit key requested
                row.append(v if not isinstance(v, dict) else "")
            writer.writerow(row)
