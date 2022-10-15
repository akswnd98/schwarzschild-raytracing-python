import numba as nb
from time import time
import numpy as np

# @nb.njit
# def f1 ():
#   a = np.array([1, 2, 3], dtype=np.float64)
#   b = np.array([1, 2 ,3], dtype=np.float64)
#   ret = 0
#   for _ in range(0, 100000000):
#     ret += np.dot(a, b)
#   return ret

# def f2 ():
#   a = np.array([1, 2, 3], dtype=np.float64)
#   b = np.array([1, 2 ,3], dtype=np.float64)
#   ret = 0
#   for _ in range(0, 100000000):
#     ret += np.dot(a, b)
#   return ret

# start = time()
# f1()
# end = time()
# print(end - start)

# start = time()
# f2()
# end = time()
# print(end - start)

print(np.array([np.array([1, 2, 3])]))