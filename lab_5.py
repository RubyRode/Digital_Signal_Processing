import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import sawtooth, square


def kern(x, A=2):
    return A * np.exp(-x ** 2)


def make_figure(r, c):
    fig, axs = plt.subplots(r, c, figsize=(7, 7))
    for x in axs:
        for ax in x:
            ax.grid(True)
            ax.set_xlim(0, 10)
    return fig, axs


class Solution:

    @staticmethod
    def sig_conv(x, kernel, A=1):
        return kernel(x, A)

    @staticmethod
    def task_1():
        lp = np.linspace(0, 10, 1000)
        sig = sawtooth(lp * np.pi / 2)
        fig, axs = make_figure(2, 2)

        axs[0, 0].plot(lp, sig)
        axs[0, 0].set_title("Original signal")

        conv = kern(lp)
        axs[0, 1].plot(lp, conv)
        axs[0, 1].set_xlim([-1, 10])
        axs[0, 1].set_title("Convolution")

        convolved = kern(sig)
        axs[1, 0].plot(lp, convolved)
        axs[1, 0].set_title("Convolved signal")

        plt.show()

    @staticmethod
    def task_3():
        lp = np.linspace(0, 2, 1000)
        sig = np.sin(lp * 5 * np.pi * 2)
        fig, axs = make_figure(2, 2)
        axs[0, 0].plot(lp, sig)
        axs[0, 0].set_xlim([0, 1])
        axs[0, 0].set_title("Original signal")

        fft_lp = fftfreq(len(lp), 1 / (len(lp) / 2))
        fft_sig = np.abs(fft(sig))
        axs[0, 1].plot(fft_lp, fft_sig)
        axs[0, 1].set_title("FFT")

        convolved = kern(sig)
        axs[1, 0].plot(lp, convolved)
        axs[1, 0].set_xlim([0, 1])
        axs[1, 0].set_title("Convolved signal")

        axs[1, 1].plot(lp, ifft(fft(sig) * kern(lp)))
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_title("IFFT of fft(signal) * convolution")

        plt.show()


s = Solution()
s.task_1()
s.task_3()
