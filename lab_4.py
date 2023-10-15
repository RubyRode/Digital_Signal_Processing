import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


#
# def draw_sign(f,filt,wc,true_sign=None,sign_lim=None):
#     N = 2000
# fmax = 2000
# T = 1/fmax
# x = np.linspace(0.0, N*T, N)
# xf = np.linspace(0.0,fmax,N)
# y = f(x)
# yf = np.fft.fft(y)
# 
# fig = plt.figure(figsize=(12,7))
# 
# plt.subplot(1,3,1)
# plt.plot(xf,np.abs(yf))
# plt.plot(xf,np.abs(yf)*filt(xf,wc))


def butter_low(w, wc):
    return (wc ** 2) / ((-w) ** 2 + (1j * np.sqrt(2) * wc * w) + wc ** 2)


def butter_high(w, wc):
    return (w ** 2) / ((-wc) ** 2 + (1j * np.sqrt(2) * wc * w) + w ** 2)


def butter_any(w, wc, n=2):
    return 1 / (1 + ((-1) ** n) * ((1j * w / wc) ** (2 * n)))


def chebyshev_n(x, n):
    if n >= 0:
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2 * x * chebyshev_n(x, n - 1) - chebyshev_n(x, n - 2)
    else:
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2 * x * chebyshev_n(x, n + 1) + chebyshev_n(x, n + 2)


def chebyshev(x, wc, n=2, eps=1):
    return 1 / (np.sqrt(1 + (eps ** 2) * (chebyshev_n(x, n) ** 2) * (x / wc)))


class Solution:

    @staticmethod
    def task_1():
        N = 2000
        fmax = 2000
        T = 1 / fmax
        x = np.linspace(0., N * T, N)
        xf = np.linspace(0., fmax, N)

        y = np.cos(2 * np.pi * 50 * x) + np.cos(2 * np.pi * 150 * x) + np.cos(2 * np.pi * 450 * x)
        yf = fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))

        for a in axs:
            for ax in a:
                ax.grid(1)

        axs[0, 0].plot(x, y)
        axs[0, 0].set_title("Original wave 50, 150, 450 Hz")
        axs[0, 1].set_xlim(0, 500)
        axs[0, 1].plot(xf, np.abs(yf))
        axs[0, 1].set_title("FFT orig signal")

        but = np.abs(butter_low(xf, 70))
        axs[1, 0].plot(xf, np.abs(yf) * but)
        axs[1, 0].set_title("Low pass filter (70 Hz)")
        axs[1, 0].set_xlim(0, 500)
        axs[1, 0].plot(xf, but * 1000)

        but = np.abs(butter_high(xf, 120))
        axs[1, 1].plot(xf, np.abs(yf) * but)
        axs[1, 1].set_title("High pass filter (120 Hz)")
        axs[1, 1].set_xlim(0, 500)
        axs[1, 1].plot(xf, but * 1000)

        plt.show()

    @staticmethod
    def task_2():
        N = 2000
        fmax = 2000
        T = 1 / fmax
        x = np.linspace(0., N * T, N)
        xf = np.linspace(0., fmax, N)

        y = np.cos(2 * np.pi * 50 * x) + np.cos(2 * np.pi * 150 * x) + np.cos(2 * np.pi * 450 * x)
        yf = fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))

        for a in axs:
            for ax in a:
                ax.set_xlim(0, 500)
                ax.grid(1)

        axs[0, 0].plot(x, y)
        axs[0, 0].set_title("Original wave 50, 150, 450 Hz")
        axs[0, 1].plot(xf, np.abs(yf))
        axs[0, 1].set_title("FFT orig signal")

        but_h = np.abs(butter_high(xf, 150))
        but_l = np.abs(butter_low(xf, 150))
        pf = but_l * but_h
        zf = but_l + but_h
        axs[1, 0].plot(xf, np.abs(yf) * pf)
        axs[1, 0].plot(xf, pf * 1000)
        axs[1, 0].set_title("PF")

        axs[1, 1].plot(xf, np.abs(yf) * zf)
        axs[1, 1].plot(xf, zf * 1000)

        plt.show()

    @staticmethod
    def task_5(noise=True):
        N = 2000
        fmax = 2000
        T = 1 / fmax
        x = np.linspace(0., N * T, N)
        xf = np.linspace(0., fmax, N)

        y = np.cos(2 * np.pi * 50 * x) + np.cos(2 * np.pi * 150 * x) + np.cos(2 * np.pi * 450 * x)
        if noise:
            y += np.random.normal(0, 1, len(x))
        yf = fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))

        for a in axs:
            for ax in a:
                ax.set_xlim(0, 500)
                ax.grid(1)

        axs[0, 0].plot(x, y)
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].set_title("Original wave 50, 150, 450 Hz")
        axs[0, 1].plot(xf, np.abs(yf))
        axs[0, 1].set_title("FFT orig signal")

        but_3 = butter_any(xf, 120, -4)
        but_4 = butter_any(xf, 120, 4)
        axs[1, 0].plot(xf, np.abs(yf) * but_3)
        axs[1, 0].plot(xf, but_3 * 1000)
        axs[1, 1].plot(xf, np.abs(yf) * but_4)
        axs[1, 1].plot(xf, but_4 * 1000)
        plt.show()


    @staticmethod
    def task_6():
        N = 2000
        fmax = 2000
        T = 1 / fmax
        x = np.linspace(0., N * T, N)
        xf = np.linspace(0., fmax, N)

        y = np.cos(2 * np.pi * 50 * x) + np.cos(2 * np.pi * 150 * x) + np.cos(2 * np.pi * 450 * x)
        yf = fft(y)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))

        for a in axs:
            for ax in a:
                ax.set_xlim(0, 500)
                ax.grid(1)

        axs[0, 0].plot(x, y)
        axs[0, 0].set_xlim(0, 1)
        axs[0, 0].set_title("Original wave 50, 150, 450 Hz")
        axs[0, 1].plot(xf, np.abs(yf))
        axs[0, 1].set_title("FFT orig signal")

        cheb = chebyshev(xf, 1000, -1, eps=1)
        axs[1, 0].plot(xf, np.abs(yf) * cheb)
        axs[1, 0].plot(xf, cheb * 1000)
        axs[1, 0].set_ylim(0, 50)
        plt.show()


if __name__ == "__main__":
    s = Solution()
    # s.task_1()
    # s.task_2()
    # s.task_5(noise=False)
    s.task_6()