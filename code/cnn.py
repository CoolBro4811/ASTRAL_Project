import os
import sys

from matplotlib import patches
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def main(fn: str):
    img = fits.open(fn)[0].data.astype(float)
    img /= img.max()  # scale 0-1
    sigma = mad_std(img)
    daofinder = DAOStarFinder(threshold=5 * sigma, fwhm=3.0)
    sources = daofinder(img)

    fig, ax = plt.subplots()

    star_coords = []
    if sources is not None:
        for row in sources:
            ax.add_patch(
                patches.Circle(
                    (row["xcentroid"], row["ycentroid"]),
                    5,
                    edgecolor="r",
                    facecolor="none",
                    linewidth=1.5,
                )
            )
            star_coords.append((int(row["xcentroid"]), int(row["ycentroid"])))

    print(f"detected: {star_coords}")

    plt.show()

    PATCH = 64
    half = PATCH // 2
    X = []
    y = []
    H, W = img.shape

    def get_patch(cx, cy):
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half
        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
            return None  # skip if off edge of img
        return img[y1:y2, x1:x2]

    for x, y in star_coords:
        patch = get_patch(x, y)
        if patch is not None:
            X.append(patch)
            y.append(1)

    rng = np.random.default_rng()
    num_neg = len(X)

    for _ in range(num_neg):
        while True:
            rx = rng.integers(half, W - half)
            ry = rng.integers(half, H - half)

            # skip if too cloes to real stars
            if all(
                (rx - sx) ** 2 + (ry - sy) ** 2 > (half**2)
                for sx, sy in star_coords
            ):
                patch = get_patch(rx, ry)
                if patch is not None:
                    X.append(patch)
                    y.append(0)
                    break
    X = np.array(X).reshape(-1, PATCH, PATCH, 1).astype(np.float32)
    y = np.array(y).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=67
    )

    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(PATCH, PATCH, 1)
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=8,
    )
    model.save("star_cnn_1.h5")

    print(f"dataset: \nX: {X.shape}\ny: {y.shape}")


if __name__ == "__main__":
    main(sys.argv[1])
