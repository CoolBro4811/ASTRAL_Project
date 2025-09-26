import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_detections(
    image,
    detections,
    aux=None,
    circle_radius=6,
    radius_mode="fixed",
    show=True,
    vmin=None,
    vmax=None,
):
    """
    Plot image with circles around detections.

    radius_mode:
      - "fixed": use circle_radius for every detection.
      - "fit": use fitted gaussian sx/sy (2*sigma) if available, otherwise fallback to circle_radius.
    aux: optional dict (from detect_stars) to also show background-subtracted / filtered panels.
    """
    n_panels = 1 + (
        1 if aux is not None else 0
    )  # main + optional combined aux view
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axs = [axs]

    im = axs[0].imshow(
        image,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title("Image")
    for d in detections:
        if radius_mode == "fit" and "fit" in d and d["fit"] is not None:
            # use average sigma and scale to radius
            sx = d["fit"].get("sx", None)
            sy = d["fit"].get("sy", None)
            if sx is not None and sy is not None:
                r = 2.0 * float(0.5 * (sx + sy))
            else:
                r = circle_radius
        else:
            r = circle_radius
        circ = Circle(
            (d["x"], d["y"]),
            radius=r,
            edgecolor="red",
            facecolor="none",
            lw=1.2,
        )
        axs[0].add_patch(circ)

    axs[0].set_xticks([])
    axs[0].set_yticks([])

    if aux is not None:
        # show background-subtracted or filtered image on second panel
        ax = axs[1]
        img_filt = aux.get("img_filt", aux.get("img_sub"))
        ax.imshow(
            img_filt, origin="lower", cmap="gray", interpolation="nearest"
        )
        ax.set_title("Filtered / Background-subtracted")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs
