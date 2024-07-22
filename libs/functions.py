import numpy as np


def ackley_cost(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)


def bukin_cost(x, y):
    term1 = 100 * np.sqrt(np.abs(y - 0.01 * x ** 2))
    term2 = 0.01 * np.abs(x + 10)
    return term1 + term2


def cross_cost(x, y):
    x1 = x
    x2 = y

    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))

    y = -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1

    return y


def drop_cost(x, y):
    numerator = 1 + np.cos(12 * np.sqrt(x ** 2 + y ** 2))
    denominator = 0.5 * (x ** 2 + y ** 2) + 2
    return -numerator / denominator


def eggholder_cost(x, y):
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(y + x / 2 + 47)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2


def griewank_cost(x, y):
    term1 = x ** 2 + y ** 2
    term2 = np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2))
    return 1 + (term1 / 4000) - term2


def holder_table_cost(x, y):
    term1 = -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi)))
    return term1


def langermann_cost(x, y):
    m = 5  # Número de termos na soma
    A = np.array([[3, 5],
                  [5, 2],
                  [2, 1],
                  [1, 4],
                  [7, 9]])
    C = np.array([1, 2, 5, 2, 3])

    sum_term = 0
    for i in range(m):
        xi = x - A[i, 0]
        yi = y - A[i, 1]
        sum_term += C[i] * np.exp(-(xi ** 2 + yi ** 2) / np.pi) * np.cos(np.pi * (xi ** 2 + yi ** 2))

    return -sum_term


def shubert_cost(x1, x2):
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        new1 = i * np.cos((i + 1) * x1 + i)
        new2 = i * np.cos((i + 1) * x2 + i)
        sum1 += new1
        sum2 += new2

    return sum1 * sum2


def levy_cost(x, y):
    xx = np.array([x, y])
    d = len(xx)
    w = 1 + (xx - 1) / 4

    term1 = (np.sin(np.pi * w[0])) ** 2
    term3 = (w[d - 1] - 1) ** 2 * (1 + 10 * (np.sin(2 * np.pi * w[d - 1])) ** 2)
    wi = w[0:(d - 1)]
    term2 = np.sum((wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * (wi + 1)))) ** 2)

    return term1 + term2 + term3


def levy_13_cost(x1, x2):
    term1 = (np.sin(3 * np.pi * x1)) ** 2
    term2 = (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
    term3 = (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)

    return term1 + term2 + term3


def rastrigin_function(x, y):
    term1 = x ** 2 - 10 * np.cos(2 * np.pi * x)
    term2 = y ** 2 - 10 * np.cos(2 * np.pi * y)
    return 2 * 10 + term1 + term2


def schaffer_2_cost(x, y):
    term1 = (np.sin(x ** 2 - y ** 2)) ** 2 - 0.5
    term2 = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
    y = 0.5 + term1 / term2
    return y


def schaffer_4_cost(x, y):
    term1 = (np.cos(np.sin(np.abs(x ** 2 - y ** 2)))) ** 2 - 0.5
    term2 = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

    y = 0.5 + term1 / term2

    return y


def schwefel_cost(x, y):
    term1 = x * np.sin(np.sqrt(abs(x)))
    term2 = y * np.sin(np.sqrt(abs(y)))
    result = 418.9829 * 2 - term1 - term2
    return result


def easom_cost(x, y):
    term1 = -np.cos(x) * np.cos(y)
    term2 = np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
    return term1 * term2


def michalewicz_cost(x, y):
    m = 10
    term1 = -np.sin(x) * (np.sin((x ** 2) / np.pi)) ** (2 * m)
    term2 = -np.sin(y) * (np.sin((2 * y ** 2) / np.pi)) ** (2 * m)
    return term1 + term2


def beale_cost(x, y):
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2
    return term1 + term2 + term3


def branin_cost(x, y):
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    term1 = a * (y - b * x ** 2 + c * x - r) ** 2
    term2 = s * (1 - t) * np.cos(x)
    return term1 + term2 + s


def goldstein_price_cost(x, y):
    term1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    term2 = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    return term1 * term2

#Função De Jong N.5
def jong_cost(x, y):
    A = np.zeros((2, 25))
    a = np.array([-32, -16, 0, 16, 32])
    A[0, :] = np.tile(a, 5)
    A[1, :] = np.repeat(a, 5)

    term1 = np.arange(1, 25)
    term2 = (x - A[0, :26])**6
    term3 = (y - A[1, :26])**6
    sum_terms = 1 / (term1 + term2 + term3)

    sum_total = np.sum(sum_terms)

    sum = 1 / (0.002 + sum_total)
    return sum
