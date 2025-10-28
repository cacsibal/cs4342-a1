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
    eigenvalues, eigenvectors = np.linalg.eig(A)
    top_k = np.argsort(np.abs(eigenvalues))[::-1][:k]
    return eigenvectors[:, top_k]

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    x = np.atleast_2d(x).reshape(-1, 1)
    return np.repeat(x, k, axis=1)

def problem9 (A):
    return A[np.random.permutation(A.shape[0])]

def problem10 (A):
    return np.mean(A, axis=1)

def problem11 (n, k):
    A = np.random.randint(0, k + 1, size=n)
    A[A % 2 == 0] = -1
    return A

def problem12 (A, b):
    b = np.atleast_2d(b).reshape(-1, 1)
    B = np.repeat(b, A.shape[1], axis=1)
    return A + B

def problem13 (A):
    n, m, _ = A.shape
    return A.reshape(n, m * m).T
