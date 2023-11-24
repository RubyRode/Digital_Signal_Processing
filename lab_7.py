import numpy as np
import matplotlib.pyplot as plt
import cmath
import scipy.io as sio
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import square
from lab_4 import butter_any
from lab_5 import plank


def dec_sampl_easy(lps, sigs, n=2):
    sig = []
    lp = []
    low = 49
    high = 51
    eps = 0.2
    for i in range(len(lps)):
        if i % n != 0:
            sig.append(sigs[i])
            lp.append(lps[i])
    sig_r, fft_ = plank(fft(sig), len(lp), low, high, eps=eps)
    return lp, np.array(sig), sig_r


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


def run_med(x, k=40):
    tmp_x = x.copy()
    return np.array([np.median(tmp_x[i - k:i + k]) for i in range(len(tmp_x))])


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
    lp = np.linspace(0, 0.5, 1000)
    # sig = np.sin(lp * 2 * np.pi * 10) + np.random.default_rng(seed=42).normal(0, 0.5, len(lp))

    sig = square(lp * 2 * np.pi * 10)

    k = 50
    og = gaussian_smooth(sig, k=k)
    rm = run_mean(sig, k=k)
    plt.plot(lp, sig)
    plt.plot(lp, rm)
    plt.plot(lp, og / og.max(), "k")
    # plt.xlim([0, 0.1])
    plt.show()


def task3():
    lp = np.linspace(0, 1, 300)
    sig = np.random.normal(loc=0, scale=0.1, size=lp.shape)
    sig[sig < 0] = 0
    sig = np.array([x / x if x != 0 else x for x in sig])

    k = 10
    ga = gaussian_smooth(sig, k=k)
    plt.bar(lp, sig, width=0.001)
    plt.plot(lp, (ga / ga.max()) - ga.min(), "r")
    plt.show()


def task4():
    lp = np.linspace(0, 1, 1000)
    sig = np.random.normal(loc=1, scale=40, size=lp.shape)
    sig[sig < -10] = 0

    k = 1
    med = run_med(sig, k=k)
    plt.bar(lp, sig, width=0.001)
    plt.plot(lp, med, "r")
    plt.xlim([0, 0.2])
    plt.show()


def task5():
    braindat = sio.loadmat(r'Lab6_Data.mat')
    time_vec = braindat['timevec'][0]
    s_rate = braindat['srate'][0]
    data = braindat['data'][0]

    first_p = data[:1008]
    second_p = data[1008:1208]
    last_p = data[1208:]

    len_window = len(second_p) // 2

    fft_first_p = fft(first_p[len(first_p) - len_window:])
    fft_first_p /= fft_first_p.max()
    fft_first_lp = fftfreq(len(fft_first_p), 1 / (len(fft_first_p) / 0.1606822400000001))

    fft_second_p = fft(last_p[:len_window])
    fft_second_p /= fft_second_p.max()
    fft_second_lp = fftfreq(len(fft_second_p), 1 / (len(fft_second_p) / 0.1606822400000001))

    plt.plot(time_vec[:1008], first_p)
    plt.plot(time_vec[1008:1208], second_p)
    plt.plot(time_vec[1208:], last_p)
    plt.show()

    plt.plot(fft_first_lp, fft_first_p)
    plt.plot(fft_second_lp, fft_second_p)
    avg_lost = run_mean(np.concatenate((fft_first_p, fft_second_p)), k=1)
    plt.plot(np.concatenate((fft_first_lp, fft_second_lp)), avg_lost)
    plt.xlim([0, 150])
    plt.show()

    found = ifft(avg_lost)
    plt.plot(time_vec[:1008], first_p / first_p.max())
    plt.plot(time_vec[1008:1208], second_p / second_p.max(), "r")
    plt.plot(time_vec[1008:1208], found / found.max())
    plt.plot(time_vec[1208:], last_p / last_p.max())
    plt.show()


def downsample(signal, n=1):
    fft_s = fft(signal)
    ny = len(fft_s) // 2 * n
    filt = butter_any(fft_s, ny, n=2) - 1
    right_n_sig = ifft(filt)[int(ny//2): int(len(filt) // 2) + int(ny)]
    lp_re = np.linspace(0, 1, len(right_n_sig))
    return lp_re, right_n_sig/right_n_sig.max()


def task6():
    lp = np.linspace(0, 1, 1000)
    sig_50 = np.sin(lp * 2 * np.pi * 50)
    sig_100 = np.sin(lp * 2 * np.pi * 400)
    sig = sig_50 + sig_100

    n_lp, n_sig, sig_r = dec_sampl_easy(lp, sig, n=2)

    fft_s = np.abs(fft(sig))
    fft_lp = fftfreq(len(sig), 1/len(sig))

    ds_flp = fftfreq(len(n_sig), 1/len(n_sig))
    fft_n_sig = np.abs(fft(n_sig))

    plt.plot(lp, sig / sig.max(), "-or")
    print(sig.shape)
    plt.plot(n_lp, n_sig / n_sig.max(), "-og")
    print(n_sig.shape)
    plt.plot(n_lp, sig_r / sig_r.max(), "-ok")
    plt.xlim([0, 0.1])
    # plt.plot(lp_re, right_n_sig / right_n_sig.max())
    # print(right_n_sig.shape)
    plt.show()
    plt.plot(ds_flp, fft_n_sig/fft_n_sig.max())
    plt.plot(fft_lp, fft_s/fft_s.max())
    plt.xlim([0, 450])
    plt.show()


def task7():
    lp_10 = np.linspace(0, 1, 1000)
    lp_35 = np.linspace(0, 1, 2000)
    lp_80 = np.linspace(0, 1, 4000)

    sig_10 = np.cos(lp_10 * 2 * np.pi * 10)
    sig_35 = np.cos(lp_35 * 2 * np.pi * 35)
    sig_80 = np.cos(lp_80 * 2 * np.pi * 80)

    fft_10 = np.abs(fft(sig_10))
    fft_lp_10 = fftfreq(len(sig_10), 1/len(sig_10))

    fft_35 = np.abs(fft(sig_35))
    fft_lp_35 = fftfreq(len(sig_35), 1/len(sig_35))

    fft_80 = np.abs(fft(sig_80))
    fft_lp_80 = fftfreq(len(sig_80), 1/len(sig_80))

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].plot(lp_10, sig_10)
    axs[0, 1].plot(fft_lp_10, fft_10)
    axs[0, 0].set_title("1000 samples")
    axs[0, 1].set_xlim([0, 100])
    axs[1, 0].plot(lp_35, sig_35)
    axs[1, 1].plot(fft_lp_35, fft_35)
    axs[1, 0].set_title("2000 samples")
    axs[1, 1].set_xlim([0, 100])
    axs[2, 0].plot(lp_80, sig_80)
    axs[2, 1].plot(fft_lp_80, fft_80)
    axs[2, 0].set_title("4000 samples")
    axs[2, 1].set_xlim([0, 100])
    plt.show()

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    ds_35 = dec_sampl_easy(lp_35, sig_35)
    ds_80 = downsample(sig_80)
    ds_80 = downsample(ds_80[1])

    fft_35 = np.abs(fft(ds_35[1]))
    fft_lp_35 = fftfreq(len(ds_35[1]), 1 / len(ds_35[1]))

    fft_80 = np.abs(fft(ds_80[1]))
    fft_lp_80 = fftfreq(len(ds_80[1]), 1 / len(ds_80[1]))

    axs[0, 0].plot(lp_10, sig_10)
    axs[0, 0].set_title(f"{len(sig_10)} samples")
    axs[0, 1].plot(fft_lp_10, fft_10)
    axs[0, 1].set_xlim([0, 100])
    axs[1, 0].plot(*ds_35)
    axs[1, 0].set_title(f"{len(ds_35[1])} samples")
    axs[1, 1].plot(fft_lp_35, fft_35)
    axs[1, 1].set_xlim([0, 100])
    axs[2, 0].plot(*ds_80)
    axs[2, 0].set_title(f"{len(ds_80[1])} samples")
    axs[2, 1].plot(fft_lp_80, fft_80)
    axs[2, 1].set_xlim([0, 100])
    plt.show()


# task1()
# task2()
# task3()
# task4()
# task5()
task6()
# task7()
