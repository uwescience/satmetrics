'''
Author: Ashley Santos

This module contains fitting and plotting functionalities.

'''
import logging

import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip


def line(x, a=0, b=0):
    return a*x+b

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


def fit(x, y, p0=None, model=gauss):
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
    if p0 is None:
        p0 = [np.max(y), ((x[-1] - x[0])/2), np.max(x)/4]
    params, unc = optim.curve_fit(model, x, y, p0=p0)

    return params


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


def plot_profile(x, y, ax=None, debug=False):
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
        if debug:
            is_streak_okay = validate_streak(a, mu, r2, nr2, x[-1], x[0], width, debug)
            ax.text(min(x), max(y), f"Validated: {is_streak_okay}")
        ax.plot(x, yhat, label=f"R2={r2:.3} \n NR2={nr2:.3}", color="red")
        ax.legend()

    ax.plot(x, y)

    return ax


def validate_streak(a, mu, r2, nr2, xmax, xmin, sigma, debug=False):
    """Validates whether the plot profile contains a streak we can successfully fit with a gaussian or not.

    Parameters
    ----------
    a : `float`
        The fit amplitude.
    mu : `float`
        The fit mean.
    r2 : `float`
        The fit root-mean-squared-deviation.
    nr2 : `float`
        The fit normalized-square-root-deviation.
    xmax : `float`
        The maximum x-value on x-axis.
    xmin : `float`
        The minimum x-value on x-axis.

    Returns
    -------
    True : `boolean`
        A confirmed streak from the image/parameters.
    False : `boolean`
        Could not confirm a streak from the image/parameters.
    """
    streak_zero_location = np.abs(mu - ((xmax - xmin) / 2))
    fwhm = 2.355 * sigma
    logging.info(f"Distance from middle: {streak_zero_location:.3}")
    logging.info(f"Full Width Half Max: {fwhm:.3}")
    logging.info(f"RMSD: {r2:.3}, NRMSD: {nr2:.3}")
    if nr2 <= 2 and streak_zero_location < 10 and fwhm < 20:
        return True
    else:
        return False


def plot_image_profile(rotated_image, ax=None, debug=False):
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

    return plot_profile(x, y, ax, debug)


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
        Factor to divide noise by.
        For example, if you set this parameter to a large value, the gaussian will have less noise added.

    Returns
    -------
    noisy_y : `numpy.array`
        Gaussian with noise added to it.
    """
    y = gauss(x, a, mu, width)
    noise = np.random.normal(size=(len(y)))/noise_level
    noisy_y = y + noise
    return noisy_y


def detrend(x, y, sigma=3, maxiters=10):
    sigma_clip = SigmaClip(sigma, maxiters)
    init_guess = [0, 0]
    a, b = fit(x, sigma_clip(y), p0=init_guess, model=line)
    return y - line(x, a, b), a, b


def fit_image(rotated_image):
    x = np.arange(0, rotated_image.shape[0], 1)
    y = list(np.median(rotated_image, axis=1))

    detrended, la, lb = detrend(x, y)

    init_guess = [np.max(y), ((x[-1] - x[0])/2), np.max(x)/4]
    try:
        a, mu, sigma = fit(x, detrended, p0=init_guess, model=gauss)
    except RuntimeError:
        return False, None, None, None, None

    yhat = gauss(x, a, mu, sigma)

    r2 = rmsd(x, detrended, yhat)
    nr2 = nrmsd(x, detrended, yhat)

    is_streak_okay = validate_streak(a, mu, r2, nr2, x[-1], x[0], sigma)

    fig, axes = plt.subplots(3, 1, figsize=(20,20), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(x, y, label="Image Data")
    ax1.plot(x, line(x, la, lb), label="Trend")
    ax1.set_title("Raw Brightness Profile")
    ax1.legend()

    ax2.plot(x, detrended, label="Data - Trend")
    ax2.legend()
    ax2.set_title("Detrended brightness profile")

    ax3.plot(x, detrended, label="Detrended signal")
    ax3.plot(x, yhat, label="Fitted Gaussian")
    ax3.set_title("Validating signal")
    ax3.legend()
    plt.savefig("validation.png")
    plt.cla()
    plt.clf()

    return is_streak_okay, a, mu, sigma, 2.355 * sigma
