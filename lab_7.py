import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import sawtooth, square
from scipy.stats import norm



def peak(x, c):
    return np.exp(-np.power(x - c, 2) / 16.0)

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return lin_interp(x, y, zero_crossings_i[0], half) - \
           lin_interp(x, y, zero_crossings_i[1], half)

def run_mean(x, k=40):
    tmp_x = x.copy()
    s = 0
    for i in range(len(tmp_x)):
        for j in range(-k, k):
            if k <= i and k <= len(x) - i:
                s += x[i + j]

        tmp_x[i] = (2 * k + 1)**-1 * s
        s = 0

    return tmp_x


def g():

def gaussian_smooth(x):


def task1():
    lp = np.linspace(0, 0.5, 10000)
    sig = np.sin(lp * 2 * np.pi * 10) + np.random.default_rng(seed=42).normal(0, 0.5, len(lp))

    plt.plot(lp, sig)
    plt.plot(lp, run_mean(sig, k=50))
    # plt.xlim([0, 0.5])
    plt.show()


task1()
