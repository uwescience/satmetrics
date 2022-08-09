'''
Author: Ashley Santos

This module contains pixel plotting module that can be used to equalize and
plot the median intensity values of the rotated image.

'''

import astropy.visualization as aviz
import numpy as np
import matplotlib.pyplot as plt


def pixelplot(rotated_image, title=""):
    """Plots a rotated equalized cutout image of a trail
    and creates a histogram of the y-axis.

    Image will be equalized using Astropy's HistEqStretch.

    Parameters
    ----------
    rotated_image : `numpy.array`
        Desired image you want to plot.
    title : `str`, optional
        Title of the main plot.

    Returns
    -------
    fig : `matplotlib.pyplot.Figure`

    ax1 : `matplotlib.pyplot.Axes`
        Axis containing the image.
    """
    left, width = 0.1, 0.65
    bottom, height = 0.1, 1
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)

    stretch = aviz.HistEqStretch(rotated_image)
    norm = aviz.ImageNormalize(rotated_image, stretch=stretch, clip=True)
    histeq = norm(rotated_image)

    h, w = rotated_image.shape
    ax.imshow(histeq, extent=(0, w, 0, h))

    bbox = ax.get_position()
    rect_histy = [bbox.xmax + spacing, bbox.ymin, 0.2, bbox.ymax - bbox.ymin]
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    ax_histy.tick_params(axis="y", labelleft=False)

    ax_histy.plot(np.median(rotated_image, axis=1), np.arange(0, h))

    return fig, ax, ax_histy
