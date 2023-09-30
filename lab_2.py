import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft, ifft, fftfreq, fft
from scipy.signal import square
from timeit import default_timer as timer


def sine(lp, frequency):
    """Returns np.ndarray with sine wave"""
    return np.sin(2 * np.pi * frequency * lp)


def cosine(lp, frequency):
    """Returns np.ndarray with sine wave"""
    return np.cos(2 * np.pi * frequency * lp)


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def my_fft(x):
    x = np.asarray(x)  # Convert input to numpy array
    N = x.shape[0]
    if N <= 1:
        return x

    even = my_fft(x[::2])  # Recursive call for evens
    odd = my_fft(x[1::2])  # Recursive call for odds

    # Calculate the FFT
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

    # Combine the even and odd parts to get the full spectrum
    spectrum = [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

    return np.array(spectrum)


class Solution:
    def __init__(self,
                 frequency=0.5,
                 linspace=np.linspace(-4, 4, 1000),
                 amplitude=1):
        self.lp = linspace
        self.A = amplitude
        self.f = frequency
        self.wv_50_hz = sine(self.lp, 50)
        self.wv_150_hz = sine(self.lp, 150)
        self.wv_sum = self.wv_150_hz + self.wv_50_hz
        self.noise_50150 = np.random.normal(0, 1, self.wv_sum.shape)

    def task_1(self):
        lps = self.lp
        start = timer()
        ff = fft(self.wv_sum)
        end = timer()
        ff_time = end - start
        start = timer()
        dft = dft_slow(self.wv_sum)
        end = timer()
        dft_time = end - start
        ht = [ff_time, dft_time, dft_time - ff_time]
        ifft_ff = ifft(ff)
        ifft_dft = ifft(dft)

        fig, axs = plt.subplots(4, 2, figsize=(9, 8))
        fig.tight_layout()
        for ax in axs:
            for i in ax:
                i.grid(1)
        axs[0, 0].set_title('Original signal')
        axs[0, 0].plot(self.lp, self.wv_sum)
        axs[1, 0].set_title('FFT of original signal')
        axs[1, 0].plot(lps, ff)
        axs[1, 1].set_title('DFT_slow of original signal')
        axs[1, 1].plot(lps, dft)
        axs[0, 1].set_title('Time comparison between fft and dft')
        axs[0, 1].bar([1, 2, 3], height=np.array(ht), width=0.2,
                      align='center', color=['b', 'g', 'r'], edgecolor='black',
                      linewidth=3, tick_label=['fft time', 'dft time', 'difference'])
        axs[2, 0].set_title('IFFT of FFT')
        axs[2, 0].plot(lps, ifft_ff)
        axs[2, 1].set_title('IFFT of DFT_slow')
        axs[2, 1].plot(lps, ifft_dft)
        axs[3, 0].plot(lps, np.abs(ifft_ff - self.wv_sum))
        axs[3, 0].set_title('Error original signal vs ifft_fft')
        axs[3, 0].set_ylim([-1, 1])
        axs[3, 1].plot(lps, np.abs(ifft_dft - self.wv_sum))
        axs[3, 1].set_title('Error original signal vs ifft_dft_slow')
        axs[3, 1].set_ylim([-1, 1])
        plt.show()
        noisy = self.wv_sum + self.noise_50150
        fft_noisy = fft(noisy)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        for ax in axs:
            for i in ax:
                i.grid(1)

        axs[0, 0].set_title('Original signal')
        axs[0, 0].plot(lps, self.wv_sum)
        axs[0, 1].set_title('Noisy original signal')
        axs[0, 1].plot(lps, noisy)
        axs[1, 0].set_title('fft of noisy signal')
        axs[1, 0].plot(lps, fft_noisy)
        axs[1, 1].set_title('ifft of fft_noisy')
        axs[1, 1].plot(lps, ifft(fft_noisy))

        plt.show()

    def task_2(self):

        lp_4 = np.linspace(0, 4, 1000)
        sq_sig = square(lp_4 * 2 * np.pi * self.f) * self.A
        noisy_sq_sig = sq_sig + np.random.normal(0, 1, sq_sig.shape)
        noisy_sq_fft = fft(noisy_sq_sig)
        sq_fft = fft(sq_sig)
        dft_sq_sig = dft_slow(sq_sig)
        dft_noisy_sq_sig = dft_slow(noisy_sq_sig)

        fig, axs = plt.subplots(3, 2, figsize=(9, 9))
        for a in axs:
            for ax in a:
                ax.grid(True)

        axs[0, 0].plot(lp_4, sq_sig)
        axs[0, 0].set_title("Original Square signal")
        axs[0, 0].set_xlim([-0.5, 4.5])
        axs[0, 0].set_ylim([-3, 3])
        axs[0, 1].plot(lp_4, noisy_sq_sig)
        axs[0, 1].set_title("Noisy Square signal")
        axs[0, 1].set_xlim([-0.5, 4.5])
        axs[0, 1].set_ylim([-6, 6])
        axs[1, 0].set_title("Discrete spectrum FFT")
        axs[1, 0].plot(lp_4, sq_fft)
        axs[1, 0].set_xlim([-0.5, 4.5])
        axs[1, 0].set_ylim([-5, 14])
        axs[2, 0].set_title("Discrete spectrum DFT_Slow")
        axs[2, 0].plot(dft_sq_sig)
        axs[1, 1].set_title("Discrete spectrum FFT noisy")
        axs[1, 1].plot(lp_4, noisy_sq_fft)
        axs[2, 1].set_title("Discrete spectrun DFT noisy")
        axs[2, 1].plot(lp_4, dft_noisy_sq_sig)
        plt.show()

    def task_3(self):
        lp_tmp = np.linspace(0, 5, 4096)
        secs = np.abs(lp_tmp.min()) + np.abs(lp_tmp.max())
        cosine_wv = np.cos(2 * np.pi * 50 * lp_tmp)
        fft_lp = fftfreq(len(lp_tmp), 1 / (len(lp_tmp) / secs))

        start = timer()
        fft_sc = fft(cosine_wv)
        end = timer()
        fft_sc_time = end - start

        start = timer()
        fft_my = my_fft(cosine_wv)
        end = timer()
        fft_my_time = end - start

        fft_ar = np.abs(fft_sc)
        my_fft_ar = np.abs(fft_my)

        mh = [fft_sc_time, fft_my_time, fft_my_time - fft_sc_time]

        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        for a in axs:
            for ax in a:
                ax.grid(True)

        axs[0, 0].plot(lp_tmp, cosine_wv)
        axs[0, 0].set_title("Cosine wave 50 Hz")

        axs[0, 1].plot(fft_lp, fft_ar)
        axs[0, 1].set_title("Cosine wave fft")
        axs[0, 1].set_xlim([0, 150])

        axs[1, 0].plot(fft_lp, my_fft_ar)
        axs[1, 0].set_title("Cosine wave my_fft")
        axs[1, 0].set_xlim([0, 150])

        axs[1, 1].set_title('Time comparison between fft and dft')
        axs[1, 1].bar([1, 2, 3], height=np.array(mh), width=0.2,
                      align='center', color=['b', 'g', 'r'], edgecolor='black',
                      linewidth=3, tick_label=['fft time', 'my_fft time', 'difference'])

        plt.show()


LENGTH = 4 * np.pi
SAMPLES = 1000
AMPLITUDE = 2
FREQUENCY = 0.5
lps = np.linspace(-LENGTH // 2, LENGTH // 2, SAMPLES)

m = Solution(FREQUENCY, lps, AMPLITUDE)
m.task_1()
m.task_2()
m.task_3()
