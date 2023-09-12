import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square
from scipy.integrate import quad
from math import *


class Ff:
    x = np.linspace(-10, 10, 1000)

    square_sig = square(x) * 2

    n = 10

    An = []
    Bn = []

    sm = 0
    T = 2

    l_cos = lambda _, x, i: square(x) * cos(i * x)
    l_sin = lambda _, x, i: square(x) * sin(i * x)

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
        self.show()

    def show(self):
        plt.plot(self.x, self.square_sig)
        plt.plot(self.x, self.sm)
        plt.show()


p = Ff()
p.forward()
