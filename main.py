import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.integrate import quad
from math import *


class Ff:

    def __init__(self, lp=np.linspace(-10, 10, 1000), amplitude=2, sig=square):
        self.x = lp
        self.A = amplitude
        self.signal = sig(self.x) * self.A
        self.func = sig

        self.n = 10

        self.An = []
        self.Bn = []
        self.error = []

        self.sm = 0
        self.T = 2

        self.l_cos = lambda x, i: self.func(x) * cos(i * x)
        self.l_sin = lambda x, i: self.func(x) * sin(i * x)

    def err(self):
        for i in range(len(self.signal)):
            self.error.append(round(self.signal[i], 5) - round(self.sm[i], 5))

    def a_n(self):
        for i in range(self.n):
            an = quad(self.l_cos, -np.pi, np.pi, args=(i,))[0] * (self.T / np.pi)
            self.An.append(an)

    def b_n(self):
        for i in range(self.n):
            bn = quad(self.l_sin, -np.pi, np.pi, args=(i,))[0] * (self.T / np.pi)
            self.Bn.append(bn)

    def main(self):
        for i in range(self.n):
            if i == 0.0:
                self.sm += self.An[i] / 2

            else:
                self.sm = self.sm + (self.An[i] * np.cos(i * self.x) + self.Bn[i] * np.sin(i * self.x))


    def forward(self):
        self.a_n()
        self.b_n()
        self.main()
        self.err()
        self.show()

    def show(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.grid(1)
        ax2.grid(1)
        ax1.plot(self.x, self.signal)
        ax1.plot(self.x, self.sm)
        ax2.plot(self.x, self.error)
        plt.show()


x = np.linspace(-10, 10, 1000)


def cosine(lp):

    try:
        cs = []
        for i in range(len(lp)):
            cs.append(sin(lp[i]))
    except TypeError as e:
        return sin(lp)
    return np.array(cs)




p = Ff(sig=square)
p.forward()
