import numpy as np

def problem1 (A, B, C):
    return A @ B - C

def problem2 (A):
    return np.ones((np.shape(A)[0], 1), dtype=int)

def problem3 (A):
    return np.identity(np.shape(A)[0], dtype=int)

def problem4 (A, i):
    return np.sum(A, axis=1)[i]

def problem5 (A, c, d):
    return np.mean(A[(A >= c) & (A <= d)])

def problem6 (A, k):
    return ...

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    return ...

def problem9 (A):
    return ...

def problem10 (A):
    return ...

def problem11 (n, k):
    return ...

def problem12 (A, b):
    return ...

def problem13 (A):
    return ...
