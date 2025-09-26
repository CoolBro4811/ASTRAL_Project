#!/usr/bin/env python3
import argparse

from src.detect import detect_stars
from src.io import load_fits_image, save_catalog_csv
from src.visual import plot_detections


def main():
    p = argparse.ArgumentParser(
        description="Run star detection on a FITS image"
    )
    p.add_argument("fits", help="FITS filename")
    p.add_argument(
        "--out", help="CSV output filename", default="detections.csv"
    )
    p.add_argument(
        "--k", type=float, default=3.0, help="detection threshold in sigma"
    )
    p.add_argument(
        "--psf_sigma",
        type=float,
        default=1.2,
        help="expected PSF sigma (pixels)",
    )
    p.add_argument("--plot", action="store_true", help="show interactive plot")
    p.add_argument(
        "--invert", action="store_true", help="Invert image before detection"
    )
    p.add_argument(
        "--radius_mode",
        choices=["fixed", "fit"],
        default="fixed",
        help="circle radius mode for plotting",
    )
    args = p.parse_args()

    img = load_fits_image(args.fits)
    detections, aux = detect_stars(
        img,
        k=args.k,
        psf_sigma=args.psf_sigma,
        min_area=2,
        invert=args.invert,
    )
    save_catalog_csv(detections, args.out)
    print(f"Saved {len(detections)} detections to {args.out}")

    if args.plot:
        plot_detections(
            img,
            detections,
            aux=aux,
            circle_radius=2 * args.psf_sigma,
            radius_mode=args.radius_mode,
        )


if __name__ == "__main__":
    main()
