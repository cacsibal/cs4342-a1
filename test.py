import numpy as np
import homework1_numpy_bgerlach_ccsibal as hw

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])
x = np.array([1, 2])

D = np.array([A, B, C])

# print(hw.problem1(A, B, C))
# print(hw.problem2(A))
# print(hw.problem3(A))
# print(hw.problem4(A, 0))
# print(hw.problem5(A, 2, 3))
# print(hw.problem10(A))
# print(hw.problem8(x, 3))
# print(hw.problem11(10, 10))
# print(hw.problem12(A, x))
print(hw.problem13(D))