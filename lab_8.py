import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import quad


def task_1():
    h = 0.1
    a = 0
    b = 1 + 0.001
    lam = 0.5
    x = np.arange(a, b, h)

    def quadro_method(K, f, a, b, h):
        x = np.arange(a, b, h)
        x = x.reshape(len(x), 1)
        n = len(x)
        wt = 1 / 2
        wj = 1
        A = np.zeros((n, n))
        for i in range(n):
            A[i][0] = -h * wt * K(x[i], x[0])
            for j in range(1, n - 1, 1):
                A[i][j] = -h * wj * K(x[i], x[j])
            A[i][n - 1] = -h * wt * K(x[i], x[n - 1])
            A[i][i] = A[i][i] + 1
        B = np.zeros((n, 1))
        for j in range(n):
            B[j][0] = f(x[j])
        y = np.linalg.solve(A, B)
        return y

    K = lambda x1, s: x1 * s * lam
    f = lambda x1: 5 / 6 * x1
    y_exact = lambda x1: x1

    y = []  # точное решение
    for i in range(len(x)):
        y.append([])  # создаем пустую строку
        y[i].append(y_exact(x[i]))
    y = np.array(y).reshape(len(x), 1)  # точное решение

    y_approx = quadro_method(K, f, a, b, h)
    plt.plot(x, y, '-g', linewidth=2, label='y_exact')  # график точного решения
    plt.plot(x, y_approx, 'or', label='y_approx')  # график найденного решения
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.ylim([0, max(y) + 2])
    plt.show()


def task_2():
    a = 0
    b = 1.001
    h = 0.05
    Lambda = -1
    x = np.arange(a, b, h)
    x = x.reshape(len(x), 1)
    n = len(x)

    f = lambda t: np.exp(t) - t
    y_exact = lambda t: 1  # точное решение

    y = []  # точное решение
    for i in range(n):
        y.append([])  # создаем пустую строку
        y[i].append(y_exact(x[i]))
    y = np.array(y).reshape(n, 1)  # точное решение

    def Solve(a, b, f, t, Lambda):

        alpha = lambda t: [-t, t, t ** 2, t ** 3, t ** 4]
        beta = lambda t: [1, 1, t, 0.5 * t ** 2, 1 / 6 * t ** 3]

        def bfun(t, m, f):
            return beta(t)[m] * f(t)

        def Aijfun(t, m, k):
            return beta(t)[m] * alpha(t)[k]

        m = len(alpha(0))  # определяем размер alpha
        M = np.zeros((m, m))
        r = np.zeros((m, 1))

        for i in range(m):
            r[i] = integrate.quad(bfun, a, b, args=(i, f))[0]
            for j in range(m):
                M[i][j] = -Lambda * integrate.quad(Aijfun, a, b, args=(i, j))[0]

        for i in range(m):
            M[i][i] = M[i][i] + 1

        c = np.linalg.solve(M, r)
        aij = np.array(alpha(t))

        return Lambda * (np.sum(c[:, np.newaxis] * aij, axis=0)) + f(t)

    y_approx = Solve(a, b, f, x, Lambda)
    plt.plot(x, y, '-g', linewidth=2, label='y_exact')  # график точного решения
    plt.plot(x, y_approx, 'or', label='y_approx')  # график найденного решения
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend('1', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.ylim([0, max(y) + 0.1])
    plt.show()


def task_3():
    phi = [
        lambda x: x,
        lambda x: x ** 2
    ]

    psi = [
        lambda x: 1,
        lambda x: x
    ]

    K = lambda x, s: (x ** 2 + x * s)
    f = lambda x: 1
    lam = 1

    def galkin_petrov(b, a, psi, K, f, lam, phi):
        for i in range(2):
            b[i] = lam * quad(lambda x: psi[i](x) * quad(lambda s: K(x, s) * f(s), -1.001, 1.001)[0], -1.001, 1.001)[0]
            for j in range(2):
                a[i][j] = quad(lambda x: phi[i](x) * psi[j](x), -1.001, 1.001)[0] - lam * \
                          quad(lambda x: psi[i](x) * quad(lambda s: K(x, s) * phi[j](s), -1.001, 1.001)[0], -1.001,
                               1.001)[
                              0]

        return a, b

    a = np.zeros([2, 2])
    b = np.zeros(2)

    a, b = galkin_petrov(a=a, b=b, psi=psi, phi=phi, K=K, lam=lam, f=f)

    x = np.linspace(-1, 1, 10)
    c = np.linalg.solve(a, b)

    plt.plot(x, 1 + 6 * x ** 2, '-g', linewidth=2, label='y_exact')
    plt.plot(x, 1 + c[0] * phi[0](x) + c[1] * phi[1](x), 'or', label='y_approx')
    plt.legend()
    plt.show()

# task_1()
# task_2()
task_3()