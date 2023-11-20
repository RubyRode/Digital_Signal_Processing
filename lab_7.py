import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import sawtooth, square
from scipy.stats import norm


def find_half_max(signal):
    max_amplitude = np.max(signal)
    half_max = max_amplitude / 2
    indices = np.where(signal >= half_max)[0]

    first_index = indices[0]
    last_index = indices[-1]

    if len(indices) % 2 == 0:
        first_index -= 1
        last_index += 1

    return last_index - first_index


def run_mean(x, k=40):
    tmp_x = x.copy()
    s = 0
    for i in range(len(tmp_x)):
        for j in range(-k, k):
            if k <= i and k <= len(x) - i:
                s += x[i + j]

        tmp_x[i] = (2 * k + 1) ** -1 * s
        s = 0

    return tmp_x


def g(mw, t):
    return np.exp((-4 * cmath.log(2, np.e) * t ** 2) / mw ** 2)


def gaussian_smooth(y, k):
    tmp_x = y.copy()
    s = 0

    mw = find_half_max(y)
    for i in range(len(tmp_x)):
        for j in range(-k, k):
            if k <= i and k <= len(y) - i:
                s += y[i + j] * g(mw, y[i + j])

        tmp_x[i] = s
        s = 0

    return tmp_x


def task1():
    lp = np.linspace(0, 0.5, 10000)
    sig = np.sin(lp * 2 * np.pi * 10) + np.random.default_rng(seed=42).normal(0, 0.5, len(lp))

    plt.plot(lp, sig)
    plt.plot(lp, run_mean(sig, k=50))
    plt.show()


def task2():
    lp = np.linspace(0, 0.5, 10000)
    sig = np.sin(lp * 2 * np.pi * 10) + np.random.default_rng(seed=42).normal(0, 0.5, len(lp))
    k = 50
    og = gaussian_smooth(sig, k=k)
    rm = run_mean(sig, k=k)
    plt.plot(lp, sig/sig.max())
    plt.plot(lp, rm / (sig.max()/rm.max()))
    plt.plot(lp, og / (og.max()/rm.max()), "k")
    plt.show()


def task3():
    lp = np.linspace(0, 0.5, 1000)
    sig = np.random.normal(loc=1, scale=0.1, size=lp.shape)
    sig[sig < 0] = 0
    # sig = np.array([x/x for x in sig])
    plt.bar(lp, sig)
    plt.show()


# task1()
# task2()
task3()