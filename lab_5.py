import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import sawtooth, square
from scipy.stats import norm


def draw_plot(signals, spectrums, fft_lp=None, xlims_signal=None, ylims_signal=None,
              xlims_spectrum=None, ylims_spectrum=None, bars=True):
    signals = np.array(signals)
    spectrums = np.array(spectrums)
    num_signals = len(signals)
    signals = signals.reshape(num_signals, -1)
    spectrums = spectrums.reshape(num_signals, -1)

    lght = len(signals)
    plt.figure(figsize=(12, 5 * lght))

    for i in range(lght):
        plt.subplot(lght, 2, (2 * i) + 1)
        plt.plot(signals[i])
        if xlims_signal is not None:
            plt.xlim(xlims_signal[0], xlims_signal[1])
        if ylims_signal is not None:
            plt.ylim(ylims_signal[0], ylims_signal[1])

        plt.subplot(lght, 2, (2 * i) + 2)
        if bars:
            plt.bar(fft_lp, np.abs(spectrums[i]))
        elif fftfreq is None:
            plt.plot(spectrums[i])
        else:
            plt.plot(fft_lp, np.abs(spectrums[i]))

        if xlims_spectrum is not None:
            plt.xlim(xlims_spectrum[0], xlims_spectrum[1])
        if ylims_spectrum is not None:
            plt.ylim(ylims_spectrum[0], ylims_spectrum[1])

    plt.tight_layout()
    plt.show()
    return None


def kern(x, A=2):
    return A * np.exp(-x ** 2)


def make_figure(r, c, limit=(0, 10)):
    fig, axs = plt.subplots(r, c, figsize=(7, 7))
    for x in axs:
        for ax in x:
            ax.grid(True)
            ax.set_xlim(*limit)
    return fig, axs


def gaussian_kernel(size, sigma):
    offset = size // 2
    kernel = np.zeros((size, size))
    for x in range(-offset, offset + 1):
        for y in range(-offset, offset + 1):
            kernel[x + offset, y + offset] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def bandpass_normal_filter(signal, freq_low, freq_high, discrete_freq):
    freqs = fftfreq(len(signal), 1 / discrete_freq)

    mu = (freq_low + freq_high) / 2
    sigma = (freq_high - freq_low) / 4
    pdf = norm.pdf(freqs, mu, sigma)
    spectrum = np.abs(fft(signal))

    filtered_spectrum = spectrum * pdf

    mask = int(np.max(spectrum) + np.min(spectrum) / np.max(filtered_spectrum))

    filtered_spectrum *= mask
    filtered_signal = ifft(filtered_spectrum)

    return filtered_signal, filtered_spectrum, spectrum, freqs


def plank(spec, n, low, high, eps=0.4):
    if not 0 < eps <= 0.5:
        raise Exception("eps must be 0 < eps <= 0.5")

    def a(k):
        if k == 0 or k == n - 1:
            return 0

        if 0 < k < eps * (n - 1):
            return 1 / (np.exp(z_a(k)) + 1)

        if eps * (n - 1) <= k <= (1 - eps) * (n - 1):
            return 1

        if (1 - eps) * (n - 1) < k < n - 1:
            return 1 / (np.exp(z_b(k)) + 1)

    def z_a(k):
        return eps * (n - 1) * (1 / k + 1 / (k - eps * (n - 1)))

    def z_b(k):
        return eps * (n - 1) * (1 / (n - 1 - k) + 1 / ((1 - eps) * (n - 1) - k))

    result_plank = [z_a(k) for k in range(low, high)]
    for i in range(n - high):
        result_plank.append(0)

    temp_plank = np.zeros(len(spec))
    temp_plank[low:len(spec)] = result_plank
    result_plank = temp_plank

    filtred_spectrum = spec * result_plank

    filtred_signal = np.fft.ifft(filtred_spectrum)

    return filtred_signal, filtred_spectrum


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
        lp = np.linspace(0, 10, 10000)
        sig = np.sin(lp * 40 * np.pi * 2)
        sig_2 = np.sin(lp * 3 * np.pi * 2)
        kern_ = kern(lp)
        fig, axs = make_figure(2, 2, limit=(0, 50))
        axs[0, 0].plot(lp, sig)
        axs[0, 0].plot(lp, sig_2)
        axs[0, 0].set_xlim([0, 0.5])
        axs[0, 0].set_title("Original signals")

        fft_lp = fftfreq(len(lp), 1 / (len(lp) / 10))
        fft_sig = np.abs(fft(sig))
        fft_sig_2 = np.abs(fft(sig_2))
        fft_kern = np.abs(fft(kern_))
        axs[0, 1].plot(fft_lp, fft_sig / fft_sig.max())
        axs[0, 1].plot(fft_lp, fft_kern / fft_kern.max())
        axs[0, 1].plot(fft_lp, fft_sig_2 / fft_sig_2.max())
        axs[0, 1].set_title("FFT")

        axs[1, 0].plot(fft_lp, fft_kern * fft_sig_2, "r")
        axs[1, 0].plot(fft_lp, fft_kern * fft_sig, "b")
        axs[1, 0].set_title("FFT after conv")

        axs[1, 1].plot(lp, ifft(fft_sig * fft_kern))
        axs[1, 1].plot(lp, ifft(fft_sig_2 * fft_kern))
        axs[1, 1].set_xlim([0, 1])
        axs[1, 1].set_title("Signals after conv")

        plt.show()

    @staticmethod
    def task_4():
        lp = np.linspace(0, 10, 1005)
        signal = np.sin(lp * 2 * np.pi * 2) + np.sin(lp * 4 * np.pi * 2) + np.sin(lp * 8 * np.pi * 2)
        spectrum = np.abs(fft(signal))
        fft_lp = fftfreq(len(lp), 1 / (len(lp) / 10))

        kern_size = 5
        sigma = 10
        kernel = gaussian_kernel(kern_size, sigma)
        # k_lp = np.linspace(0, 20, 25)
        # kern_l = np.convolve(lp, kernel.flatten(), mode='same')
        kern_l = kern(lp)
        blurred_sig = np.convolve(signal, kernel.flatten(), mode='same')
        blurred_sig_spectrum = np.abs(fft(blurred_sig))

        fig, axs = make_figure(2, 2, limit=(0, 10))

        axs[0, 0].plot(lp, signal)
        axs[0, 0].set_xlim([0, 1])
        axs[0, 1].plot(fft_lp, spectrum)
        axs[1, 0].plot(lp, blurred_sig)
        axs[1, 0].set_xlim([0, 1])
        axs[1, 1].plot(fft_lp, blurred_sig_spectrum / blurred_sig_spectrum.max())
        axs[1, 1].plot(fft_lp, np.abs(fft(kern_l)) / 17.5)
        # axs[1, 1].set_xlim([0, 20])

        plt.show()

    @staticmethod
    def task_5():
        lp = np.linspace(0, 10, 1500)
        signal = np.sin(lp * 2 * np.pi * 2) + np.sin(lp * 10 * np.pi * 2) + np.sin(lp * 15 * np.pi * 2)
        filt_sig_bp, filt_spec_bp, spec, fft_lp = bandpass_normal_filter(signal, 2, 11, len(lp) / 10)
        filt_sig_lp, filt_spec_lp, _, _ = bandpass_normal_filter(signal, 0, 20, len(lp) / 10)

        signals = [signal, filt_sig_bp, filt_sig_lp]
        spectrums = [spec, filt_spec_bp, filt_spec_lp]

        draw_plot(signals, spectrums, fft_lp, xlims_signal=(0, 1500),
                  xlims_spectrum=(0, 25), bars=False)

    @staticmethod
    def task_6():
        eps = 0.2
        low = 5
        high = 10
        lp = np.linspace(0, 1, 100000)
        signal = np.sin(lp * 2 * np.pi * 2) + np.sin(lp * 10 * np.pi * 2) + np.sin(lp * 15 * np.pi * 2)
        spec = fft(signal)
        fft_lp = fftfreq(len(lp), 1 / len(lp))

        filt_sig, filt_spec = plank(spec, len(lp), low, high, eps=eps)

        fl_lp, fl_sp = plank(lp, len(lp), low, high, eps=eps)

        signals = [signal, filt_sig, fl_lp]
        specs = [spec, filt_spec, fl_sp]

        draw_plot(signals, specs, fft_lp=fft_lp, xlims_spectrum=(0, 25), bars=False)

    @staticmethod
    def task_7():
        eps = 0.2
        low = 14
        high = 16
        lp = np.linspace(0, 1, 1000)
        signal = np.sin(lp * 2 * np.pi * 2) + np.sin(lp * 10 * np.pi * 2) + np.sin(lp * 15 * np.pi * 2)
        spec = fft(signal)
        fft_lp = fftfreq(len(lp), 1 / len(lp))

        signal += np.random.normal(size=signal.shape, loc=30, scale=3) / 10

        plank_sig, plank_spec = plank(spec, len(lp), low, high, eps=eps)
        norm_sig, norm_spec, _, _ = bandpass_normal_filter(signal, low, high, len(lp))

        sigs = [signal, norm_sig, plank_sig]
        specs = [signal, norm_spec, plank_spec]

        draw_plot(sigs, specs, fft_lp=fft_lp, bars=False, xlims_spectrum=(0, 25))


s = Solution()
s.task_1()
s.task_3()
s.task_4()
s.task_5()
s.task_6()
s.task_7()
