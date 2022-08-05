'''
Author: Ashley Santos

This module contains fitting and plotting functionalities.

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim


def gauss(x, a, mu, width):
    """Calculates a gaussian in sample points with the given parameters.

    Parameters
    ----------
    x : `numpy.array`
        Range of values that gauss is applied on.
    a : `float`
        The amplitude.
    mu : `float`
        The mean.
    width : `float`
        The standard deviation.

    Returns
    -------
    y : `numpy.array`
        The y-values of the gaussian line.
    """
    return a*np.exp(-(((x-mu)/(2*width))**2))


def fit(x, y):
    """Fits a gaussian to the given sample.

    Parameters
    ----------
    x : `numpy.array`
        X coordinates of the sample.
    y : `numpy.array`
        Y coordinates of the sample.

    Returns
    -------
    a : `float`
        The amplitude.
    mu : `float`
        The mean.
    width : `float`
        The standard deviation.
    """
    (a, mu, width), unc = optim.curve_fit(gauss, x, y, p0=[200, ((x[-1] - x[0])/2), 50])

    return a, mu, width


def rmsd(x, y, yhat):
    """Calculates root-mean-square deviation of the given parameters.

    Parameters
    ----------
    x : `numpy.array`
        X coordinates of the sample.
    y : `numpy.array`
        Y coordinates of the sample.
    yhat : `numpy.array`
        Predicted Y coordinates of the sample.

    Returns
    -------
    rmsd: `float`
        Root-mean-square deviation.
    """
    return np.sqrt(np.sum((yhat - y) ** 2 / len(y)))


def nrmsd(x, y, yhat):
    """Calculates normalized root-mean-square deviation of the given parameters.

    RMSD is normalized using the mean.

    Parameters
    ----------
    x : `numpy.array`
        X coordinates of the sample.
    y : `numpy.array`
        Y coordinates of the sample.
    yhat : `numpy.array`
        Predicted Y coordinates of the sample.

    Returns
    -------
    nrmsd: `float`
        Normalized root-mean-square deviation.
    """
    nrmsd = rmsd(x, y, yhat)
    return nrmsd/np.mean(y)


def plot_profile(x, y, ax=None):
    """Plots the profile of given sample coordinates.

    Parameters
    ----------
    x : `numpy.array`
        X coordinates of the sample.
    y : `numpy.array`
        Y coordinates of the sample.
    ax : `matplotlib.pyplot.Axis`
        The axis the data should be plotted on.

    Returns
    -------
    ax : `matplotlib.pyplot.Axis`
        The axis the data should be plotted on.
    """
    if ax is None:
        fig, ax = plt.subplots()

    try:
        a, mu, width = fit(x, y)
        fit_success = True
    except RuntimeError:
        fit_success = False

    if fit_success:
        yhat = gauss(x, a, mu, width)
        r2 = rmsd(x, y, yhat)
        nr2 = nrmsd(x, y, yhat)
        ax.plot(x, yhat, label=f"R2={r2:.3} \n NR2={nr2:.3}", color="red")
        ax.legend()

    ax.plot(x, y)

    return ax


def plot_image_profile(rotated_image, ax=None):
    """Plots the profile of given image array.

    Parameters
    ----------
    rotated_image : `numpy.array`
        Desired image you want to plot.
    ax : `matplotlib.pyplot.Axis`
        The axis the data should be plotted on.

    Returns
    -------
    ax : `matplotlib.pyplot.Axis`
        The axis the data should be plotted on.
    """
    x = np.arange(0, rotated_image.shape[0], 1)
    y = list(np.median(rotated_image, axis=1))

    return plot_profile(x, y, ax)


def generate_data(x, a, mu, width, noise_level=10):
    """Given sample coordinates with generate data with random noise, which can be adjusted with the parameters.

    Parameters
    ----------
    x : `numpy.array`
        X coordinates of the sample.
    a : `float`
        The amplitude.
    mu : `float`
        The mean.
    width : `float`
        The standard deviation.
    noise_level : `float`, optional
        Given amount of noise to add to normal distribution.

    Returns
    -------
    noisy_y : `numpy.array`
        Gaussian with noise added to it.
    """
    y = gauss(x, a, mu, width)
    noise = np.random.normal(size=(len(y)))/noise_level
    noisy_y = y + noise
    return noisy_y
