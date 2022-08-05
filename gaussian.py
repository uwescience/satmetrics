import line_detection
import image_rotation
from pixelplot import pixelplot
from astropy.io import fits
import astropy.visualization as aviz
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim


def gauss(x, a, mu, width):
    return a*np.exp(-(((x-mu)/(2*width))**2))

def fit(x,y):
    (a, mu, width), unc = optim.curve_fit(gauss, x, y, p0=[200, ((x[-1] - x[0])/2), 50])

    return a, mu, width

def rmsd(x,y, yhat):
    #yhat = fit(x,y)
    return np.sqrt(np.sum((yhat - y)**2/ len(y)))

def nrmsd(x,y, yhat):
    nrmsd = rmsd(x, y, yhat)
    return nrmsd/np.mean(y)

def plot_profile(x,y, ax=None):
    a, mu, width = fit(x,y)
    yhat = gauss(x, a, mu, width)
    r2 = rmsd(x, y, yhat)
    nr2 = nrmsd(x, y, yhat)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x,y)
    ax.plot(x, gauss(x, a, mu, width), label = f"R2={r2:.3} \n NR2={nr2:.3}")
    ax.legend()

    return ax

def plot_image_profile(rotated_image, ax=None):
    x = np.arange(0, rotated_image.shape[0], 1)
    y = list(np.median(rotated_image, axis = 1))

    return plot_profile(x,y,ax)
    
def generate_data(a, mu, width, noise_level=10, xlim=(-10,10), step=0.5):
    x = np.arange(xlim[0], xlim[1], step)
    y = gauss(x, a, mu, width)
    noise = np.random.normal(size=(len(y)))/noise_level
    return y + noise

def confirm_streak(a, y):
    if max(y) * 0.1 >= a - max(y):
        Test = True
    else: 
        Test = False
    print(Test) #use rmse , create a couple of examples add mroe noise and see when its certain its not a trail anymore
    

