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




# import numpy as np
# import matplotlib.pyplot as plt
# import math as m
# import scipy.integrate as itg
# import scipy.special as spec
#
#
# def x_(t_0):
#     res = 0
#     for k in range(1, 10, 2):
#         res += (m.sin(k * 2 * m.pi * t_0 / 2)) / k
#     res = (4 * 2 * res) / m.pi
#     return res
#
#
# class Equation:
#
#     def __init__(self, N=10):
#         self.n = 1
#         self.N = N
#         self.linsp = np.linspace(-4, 4, 800)
#
#     def func_show(self, func):
#         sig = np.zeros_like(self.linsp)
#         for i in range(1, len(self.linsp)):
#             sig[i] = func(self.linsp[i])
#
#         return self.linsp, sig
#
#     def main(self, t_0):
#         a0 = self.a_0_(t_0)
#         res = 0
#         for i in range(self.n, self.N):
#             res += self.a_n_(t_0) * m.cos(self.n * (2 * np.pi / 2) * t_0) + \
#                    self.b_n_(t_0) * m.sin(self.n * (2 * np.pi / 2) * t_0)
#
#         return (a0 / 2) + res
#
#     def x_cos(self, t_0):
#         return x_(t_0) * m.cos(self.n * 2 * np.pi / 2 * t_0)
#
#     def x_sin(self, t_0):
#         return x_(t_0) * m.sin(self.n * 2 * np.pi / 2 * t_0)
#
#     def a_0_(self, t_0):
#         res = 0
#
#         it = itg.quad(x_, t_0, t_0 + 2)
#         res += it[0]
#         return res
#
#     def a_n_(self, t_0):
#         res = 0
#
#         it = itg.quad(self.x_cos, t_0, t_0 + 2)
#         res += it[0]
#         return res
#
#     def b_n_(self, t_0):
#         res = 0
#
#         it = itg.quad(self.x_sin, t_0, t_0 + 2)
#         res += it[0]
#         return res
#
#
# Amplitude = 2
# T = 2
#
# eq = Equation()
#
# t, signal = eq.func_show(x_)
# t_m, signal_m = eq.func_show(eq.main)
# plt.plot(t, signal)
# plt.plot(t_m, signal_m)
#
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
#
# plt.grid(True)
# plt.show()
