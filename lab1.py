import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
from scipy.integrate import quad
from math import *
from scipy.fft import rfft, rfftfreq


class Ff:

    def __init__(self, f=0.5, lp=np.linspace(-4 * np.pi, 4 * np.pi, 1000), amplitude=2, sig=square):
        self.x = lp
        self.A = amplitude
        self.f = f
        self.secs = np.abs(self.x.min()) + np.abs(self.x.max())
        self.T = 1 / self.f
        self.signal = sig(self.x * (2 * np.pi / self.T)) * self.A
        self.func = sig
        self.fft_ar = []
        self.fft_lp = []
        self.fft_or = []
        self.fft_n = []
        self.noise = np.random.normal(0, 0.1, len(self.x))
        self.n = 10
        self.An = []
        self.Bn = []
        self.error = []
        self.sm = 0
        self.l_cos = lambda k, i: self.func((2 * np.pi / self.T) * k) * cos(i * (2 * np.pi / self.T) * k)
        self.l_sin = lambda k, i: self.func((2 * np.pi / self.T) * k) * sin(i * (2 * np.pi / self.T) * k)

    def err(self):
        """Error x(t) - x*(t)"""
        for i in range(len(self.signal)):
            self.error.append(round(self.signal[i], 5) - round(self.sm[i], 5))

    def a_n(self):
        """An coefficient"""
        for i in range(self.n):
            an = quad(self.l_cos, -np.pi, np.pi, args=(i,))[0] * (self.T / np.pi)
            self.An.append(an)

    def b_n(self):
        """Bn coefficient"""
        for i in range(self.n):
            bn = quad(self.l_sin, -np.pi, np.pi, args=(i,))[0] * (self.T / np.pi)
            self.Bn.append(bn)

    def ff_t(self):
        """FFT check"""
        self.fft_lp = rfftfreq(len(self.x), 1 / (len(self.x) / self.secs))
        self.fft_ar = rfft(self.sm)
        self.fft_or = rfft(self.signal)
        self.fft_n = rfft(self.sm + self.noise)

    def main(self):
        """Main function for Fourier series"""
        for i in range(self.n):
            if i == 0.0:
                self.sm += self.An[i] / 2 * self.A / 2

            else:
                self.sm += (self.An[i] * np.cos(i * (2 * np.pi / self.T) * self.x) +
                            self.Bn[i] * np.sin(i * (2 * np.pi / self.T) * self.x)) * self.A / 2

    def forward(self):
        """Forward pass. Computes An, Bn, x*(t), error, scipy.fft check and shows a plot"""
        self.a_n()
        self.b_n()
        self.main()
        self.err()
        self.ff_t()
        self.show()

    def show(self):
        """Plot setup + show"""
        fig,  ax = plt.subplots(nrows=5, ncols=1)
        (ax1, ax2, ax3, ax4, ax5) = ax
        fig.set_figheight(8)
        for i in ax:
            i.grid(1)
        ax2.set_ylim([-self.A - 1, self.A + 1])
        ax1.set_ylim([-self.A - 1, self.A + 1])
        ax3.set_xlim([0, self.f + self.f / 2])
        ax4.set_xlim([0, self.f + self.f / 2])
        ax1.plot(self.x, self.signal, 'b')
        ax1.plot(self.x, self.sm, 'r')
        ax2.plot(self.x, self.error, 'g')
        ax3.plot(self.fft_lp, np.abs(self.fft_ar), 'r')
        ax4.plot(self.fft_lp, np.abs(self.fft_or), 'b')
        ax5.plot(self.fft_lp, np.abs(self.fft_n), 'g')
        plt.show()


x = np.linspace(-2, 2, 1000)


def cosine(lp):
    """Cos function for the task"""
    try:
        cs = []
        for i in range(len(lp)):
            cs.append(cos(lp[i]))
    except TypeError as e:
        return cos(lp)
    return np.array(cs)


p = Ff(sig=cosine, amplitude=7, f=100, lp=x)  # Try using scipy.signal.square or scipy.signal.sawtooth
p.forward()
