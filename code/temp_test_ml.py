import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.stats import mad_std

model = tf.keras.models.load_model("star_cnn_1.h5")
img = fits(sys.argv[1])


def detect_stars_cnn(img, window=64, stride=16, threshold=0.9):
    H, W = img.shape
    detections = []

    for y in range(0, H - window, stride):
        for x in range(0, W - window, stride):
            patch = img[y : y + window, x : x + window]
            patch = patch.reshape(1, window, window, 1)

            p = model.predict(patch, verbose=0)[0][0]
            if p > threshold:
                detections.append((x + window // 2, y + window // 2, p))

    return detections


cnn_dets = detect_stars_cnn(img)
print("CNN detections:", len(cnn_dets))

img_color = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

for x, y, p in cnn_dets:
    cv2.circle(img_color, (x, y), 5, (0, 255, 0), 1)

plt.figure(figsize=(10, 10))
plt.imshow(img_color[:, :, ::-1])
plt.title("CNN Star Detections")
plt.axis("off")
plt.show()
