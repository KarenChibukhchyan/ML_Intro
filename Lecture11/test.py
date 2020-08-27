import numpy as np


def rotate_clockwise(matrix):
    N = len(matrix)
    for i in range(N//2):
        a = matrix[i, i:N - i].copy()
        b = matrix[i:N - i, N - i - 1].copy()
        c = matrix[N - i - 1, i:N - i].copy()
        d = matrix[i:N - i, i].copy()

        matrix[i, i:N - i] = d[::-1]
        matrix[i:N - i, N - i - 1] = a
        matrix[N - i - 1, i:N - i] = b[::-1]
        matrix[i:N - i, i] = c



a = np.random.randint(20, size=(7, 7))
print(a)
print()
rotate_clockwise(a)
print()
print(a)
