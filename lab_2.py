import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft, ifft
from timeit import default_timer as timer


def sine(lp, frequency):
    sine_wv = []
    for i in range(len(lp)):
        sine_wv.append(math.sin(2 * np.pi * frequency * lp[i]))
    return np.array(sine_wv)


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


LENGTH = 4 * np.pi
SAMPLES = 1000

lps = np.linspace(-LENGTH//2, LENGTH//2, SAMPLES)

wv_50 = sine(lps, 50)
wv_150 = sine(lps, 150)
wv_sum = wv_50 + wv_150
fig, axs = plt.subplots(4, 2, figsize=(9, 8))
fig.tight_layout()
for ax in axs:
    for i in ax:
        i.grid(1)

axs[0, 0].set_title('Original signal')
axs[0, 0].plot(lps, wv_sum)

start = timer()
ff = fft(wv_sum)
end = timer()
ff_time = end - start

start = timer()
dft = dft_slow(wv_sum)
end = timer()
dft_time = end - start

print(ff_time, dft_time, dft_time - ff_time)

ht = [ff_time, dft_time, dft_time - ff_time]
axs[1, 0].set_title('FFT of original signal')
axs[1, 0].plot(lps, ff)
axs[1, 1].set_title('DFT_slow of original signal')
axs[1, 1].plot(lps, dft)
axs[0, 1].set_title('Time comparison between fft and dft')
axs[0, 1].bar([1, 2, 3], height=np.array(ht), width=0.2,
           align='center', color=['b', 'g', 'r'], edgecolor='black',
           linewidth=3, tick_label=['fft time', 'dft time', 'difference'])
axs[2, 0].set_title('IFFT of FFT')
ifft_ff = ifft(ff)
axs[2, 0].plot(lps, ifft_ff)
axs[2, 1].set_title('IFFT of DFT_slow')
ifft_dft = ifft(dft)
axs[2, 1].plot(lps, ifft_dft)
axs[3, 0].plot(lps, np.abs(ifft_ff - wv_sum))
axs[3, 0].set_title('Error original signal vs ifft_fft')
axs[3, 0].set_ylim([-1, 1])
axs[3, 1].plot(lps, np.abs(ifft_dft - wv_sum))
axs[3, 1].set_title('Error original signal vs ifft_dft_slow')
axs[3, 1].set_ylim([-1, 1])
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(7, 7))

for ax in axs:
    for i in ax:
        i.grid(1)


axs[0, 0].set_title('Original signal')
axs[0, 0].plot(lps, wv_sum)
axs[0, 1].set_title('Noisy original signal')
axs[0, 1].plot(lps, wv_sum + np.random.normal(0, 1, wv_sum.shape))
plt.show()
