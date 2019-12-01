import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

def get_mat(n):
    """
    Генерация входных матриц.
    Проверяем, является ли n степенью двойки,
    если нет - достраиваем матрицу до нужного
    размера нулевыми столбцами и строками.
    :param n: размер матрицы
    :return: res - целочисленная матрица
    """
    res = np.random.randint(0, 51, (n, n))
    if np.modf(np.log2(n))[0] != 0:
        add = int((2 ** (np.floor(np.log2(n)) + 1)) - n)
        zeros = np.zeros((n + add, n + add), dtype=int)
        zeros[:n, :n] = res
        res = zeros
    return res

def get_submat(X):
    """
    Выделение подматриц
    :param X: матрица
    :return: кортеж подматриц
    """
    n = np.shape(X)[0] // 2
    a11 = X[:n, :n]
    a12 = X[:n, n:]
    a21 = X[n:, :n]
    a22 = X[n:, n:]
    return a11, a12, a21, a22


def strassen(A, B):
    """
    Имлементация алгоритма Штрассена
    :param A: первый множитель
    :param B: второй множитель
    :return: С -  результат умножения
    """

    n = np.shape(A)[0]
    C = np.zeros_like(A)

    if n == 1:
        C[0][0] = A[0][0] * B[0][0]
        return C
    else:

        k = n // 2

        A11, A12, A21, A22 = get_submat(A)
        B11, B12, B21, B22 = get_submat(B)

        S1 = B12 - B22
        S2 = A11 + A12
        S3 = A21 + A22
        S4 = B21 - B11
        S5 = A11 + A22
        S6 = B11 + B22
        S7 = A12 - A22
        S8 = B21 + B22
        S9 = A11 - A21
        S10 = B11 + B12

        P1 = strassen(A11, S1)
        P2 = strassen(S2, B22)
        P3 = strassen(S3, B11)
        P4 = strassen(A22, S4)
        P5 = strassen(S5, S6)
        P6 = strassen(S7, S8)
        P7 = strassen(S9, S10)

        C[:k, :k] = P5 + P4 - P2 + P6
        C[:k, k:] = P1 + P2
        C[k:, :k] = P3 + P4
        C[k:, k:] = P5 + P1 - P3 - P7

        return C

def wrapper():
    time_history = []
    orders = [2**i for i in range(2, 8)]
    for i in orders:
        A, B = get_mat(i), get_mat(i)
        since = time.time()
        strassen(A, B)
        time_elapsed = (time.time() - since)
        time_history.append(time_elapsed)
    return orders, time_history




