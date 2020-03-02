#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import time
import math
import numpy as np

x = [i * 0.001 for i in np.arange(1000000)]
start = time.process_time()
for i, t in enumerate(x):
    x[i] = math.sin(t)
end = time.process_time()
print("math.sin:", str(1000 * (end - start)) + "ms")

x = [i * 0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.process_time()
np.sin(x)
end = time.process_time()
print("numpy.sin:", str(1000 * (end - start)) + "ms")

x1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
# 循环计算向量点积
start = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
end = time.process_time()
print("dot = " + str(dot) + ", loop computation time: " +
      str(1000 * (end - start)) + "ms")

# Numpy dot函数计算向量点积
start = time.process_time()
dot = np.dot(x1, x2)
end = time.process_time()
print("dot = " + str(dot) + ", numpy.dot computation time: " +
      str(1000 * (end - start)) + "ms")
