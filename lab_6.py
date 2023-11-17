import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import chirp
import scipy.io as sio


def morlet(t, alpha=0.8, f=1):
    return np.exp(-t ** 2 / (alpha ** 2)) * np.exp(1j * 2 * np.pi * t * f)


def mex_hat(t):
    return (1 - t ** 2) * np.exp(-(t ** 2 / 2))


def haar_(t):
    if 0 <= t < 0.5:
        return 1
    elif 0.5 <= t < 1:
        return -1
    else:
        return 0


def haar(t):
    return np.array(list(map(haar_, t)))


def task1():
    lp = np.linspace(-3, 3, 1000)
    time_domain = morlet(lp, alpha=0.8)
    fd_lp = fftfreq(len(lp), 1 / (len(lp) / (abs(lp.min()) + abs(lp.max()))))
    freq_domain = np.abs(fft(time_domain))

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    axs[0].plot(lp, time_domain)
    axs[1].plot(fd_lp, freq_domain)
    axs[1].set_xlim([0, 20])

    plt.show()


def task2():
    lp = np.linspace(-5, 5, 1000)
    time_domain = mex_hat(lp)
    fd_lp = fftfreq(len(lp), 1 / (len(lp) / (abs(lp.min()) + abs(lp.max()))))
    freq_domain = np.abs(fft(time_domain))

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    axs[0].plot(lp, time_domain)
    axs[1].plot(fd_lp, freq_domain)
    axs[1].set_xlim([0, 20])

    plt.show()


def task3():
    lp = np.linspace(-3, 3, 1000)
    time_domain = haar(lp)
    fd_lp = fftfreq(len(lp), 1 / (len(lp) / (abs(lp.min()) + abs(lp.max()))))
    freq_domain = np.abs(fft(time_domain))

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    axs[0].plot(lp, time_domain)
    axs[1].plot(fd_lp, freq_domain)
    axs[1].set_xlim([0, 20])

    plt.show()


def task4():
    lp = np.linspace(-5, 5, 1000)
    signal = np.sin(lp * 2 * np.pi * 10) + np.random.default_rng(seed=42).normal(0, 0.2, len(lp))
    fd_lp = fftfreq(len(lp), 1 / (len(lp) / (abs(lp.min()) + abs(lp.max()))))
    freq = np.abs(fft(signal))

    fig, axs = plt.subplots(3, 2, figsize=(7, 7))

    axs[0, 0].plot(lp, signal / signal.max())
    axs[0, 1].plot(fd_lp, freq / freq.max())
    axs[0, 1].set_xlim([0, 20])
    mor = np.convolve(signal, morlet(lp, alpha=0.8, f=10), mode="same")
    axs[1, 0].plot(lp, mor / mor.max())
    axs[1, 0].set_xlim([0, 1])
    har = np.convolve(signal, haar(lp), mode="same")
    axs[1, 1].plot(lp, har / har.max())
    fd_mor = np.abs(fft(mor))
    fd_har = np.abs(fft(har))
    axs[2, 0].plot(fd_lp, fd_mor)
    axs[2, 1].plot(fd_lp, fd_har)
    axs[2, 0].set_xlim([0, 20])
    axs[2, 1].set_xlim([0, 20])

    plt.show()


def task4_b():
    discrete_freq = 1000000
    freq_wavelets = 10

    lp = np.linspace(0, 100, discrete_freq)
    m_lp = np.linspace(0, 4, freq_wavelets)
    signal = chirp(lp, f0=20, f1=1000, t1=20, method="quadratic")
    # signal = np.sin(lp * 2 * np.pi * 1000)

    sig_ax = plt.axes((0.1, 0.55, 0.85, 0.4))
    spec_ax = plt.axes((0.1, 0.085, 0.85, 0.4))

    sig_ax.plot(lp, signal)
    sig_ax.set_xlim([63.50, 64.150])

    mor = np.convolve(signal, mex_hat(m_lp), mode="same")

    spec_ax.specgram(mor, Fs=discrete_freq / 100, cmap="inferno",
                     sides='onesided')
    spec_ax.set_ylim([0, 2000])
    spec_ax.scatter(20, 1000, c='b')
    spec_ax.plot([0, 100], [1000, 1000], '-k')
    spec_ax.plot([20, 20], [0, 2000], '-k')
    spec_ax.plot([63.50, 63.50], [0, 2000], '-b')
    spec_ax.plot([64.150, 64.150], [0, 2000], '-b')
    spec_ax.set_xlim([0, 100])
    plt.show()


def task5():
    braindat = sio.loadmat(r'Lab6_Data.mat')
    time_vec = braindat['timevec'][0]
    s_rate = braindat['srate'][0]
    data = braindat['data'][0]

    discrete_freq = 50

    signal = np.linspace(8, 70, discrete_freq)
    time = np.arange(-1, 1, 1 / s_rate)

    y = ([morlet(t, 100) for t in time_vec])
    y = np.convolve(data, y)

    lst_signal = []

    for cnt in range(discrete_freq):
        lst_signal.append(np.exp(1j * 2 * np.pi * signal[cnt] * time) * np.exp(-(4 * np.log(2) * time ** 2) / 0.2 ** 2))

    dataX = fft(y, len(time_vec) + len(time) - 1)

    tf = []

    for sign in lst_signal:
        waveX = fft(sign, len(time_vec) + len(time) - 1)
        waveX = waveX / np.max(waveX)

        conv_res = ifft(waveX * dataX)
        conv_res = conv_res[len(time) // 2 - 1: -len(time) // 2]
        tf.append(np.abs(conv_res) ** 2)

    np.meshgrid()

    plt.pcolormesh(time_vec, signal, tf, vmin=0, vmax=1e3, cmap='gist_heat')
    plt.show()


def task5_b():
    y, sr = librosa.load("78143.mp3")
    y = y[:100000]
    spec = librosa.feature.melspectrogram(y=np.array(y), sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    wvlts_lp = np.linspace(0, 2, int(sr))

    mor = np.convolve(y, np.array(morlet(wvlts_lp), dtype=np.float32), mode="same")
    mor = librosa.feature.melspectrogram(y=mor, sr=sr)
    mor_db = librosa.power_to_db(mor, ref=np.max)

    hat = np.convolve(y, mex_hat(wvlts_lp), mode="same")
    hat = librosa.feature.melspectrogram(y=hat, sr=sr)
    hat_db = librosa.power_to_db(hat, ref=np.max)

    har = np.convolve(y, haar(wvlts_lp), mode="same")
    har = librosa.feature.melspectrogram(y=har, sr=sr)
    har_db = librosa.power_to_db(har, ref=np.max)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    img = librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', ax=ax[0, 0])
    mor_img = librosa.display.specshow(mor_db, x_axis='time', y_axis='mel', ax=ax[0, 1])
    hat_img = librosa.display.specshow(hat_db, x_axis='time', y_axis='mel', ax=ax[1, 0])
    har_img = librosa.display.specshow(har_db, x_axis='time', y_axis='mel', ax=ax[1, 1])
    # ax[0, 0].colorbar(img, ax=ax[0, 0], format='%+2.0f dB')
    ax[0, 0].set_title('Spectrogram')

    # ax[0, 1].colorbar(mor_img, ax=ax[0, 1], format='%+2.0f dB')
    ax[0, 1].set_title('Morlet')

    # ax[1, 0].colorbar(hat_img, ax=ax[1, 0], format='%+2.0f dB')
    ax[1, 0].set_title('Mexican hat')

    # ax[1, 1].colorbar(har_img, ax=ax[1, 1], format='%+2.0f dB')
    ax[1, 1].set_title('Haar')

    plt.show()


# task1()
# task2()
# task3()
# task4()
# task4_b()
# task5()
# task5_b()
