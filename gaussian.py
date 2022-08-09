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
        is_streak_okay = validate_streak(a, mu, r2, nr2, x[-1], x[0])
        if is_streak_okay:
            print("Gaussian fit looks reasonable.")
            ax.plot(x, yhat, label=f"R2={r2:.3} \n NR2={nr2:.3}", color="red")
            ax.legend()
        else:
            print("Could not fit gaussian to streak profile.")

    ax.plot(x, y)

    return ax


def validate_streak(a, mu, r2, nr2, xmax, xmin, sigma):
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
    if r2 <= 0.1 and nr2 <= 2 and streak_zero_location < 10 and fwhm < 20:
        return True
    else:
        return False


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


def fit_image(rotated_image):
    x = np.arange(0, rotated_image.shape[0], 1)
    y = list(np.median(rotated_image, axis=1))

    try:
        a, mu, sigma = fit(x, y)
    except RuntimeError:
        return False, None, None, None, None

    yhat = gauss(x, a, mu, sigma)

    r2 = rmsd(x, y, yhat)
    nr2 = nrmsd(x, y, yhat)

    is_streak_okay = validate_streak(a, mu, r2, nr2, x[-1], x[0])

    return is_streak_okay, a, mu, sigma, 2.355 * sigma