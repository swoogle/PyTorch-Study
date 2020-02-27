#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import numpy as np

list1 = [3.14, 2.17, 0, 1, 2]
nd1 = np.array(list1)
print("---- nd1 ----")
print(nd1)

list2 = [[3.14, 2.17, 0, 1, 2], [1, 2, 3, 4, 5]]
nd2 = np.array(list2)
print("---- nd2 ----")
print(nd2)

nd3 = np.random.random([3, 3])
print("---- nd3 ----")
print(nd3)

np.random.seed(123)
nd4 = np.random.randn(2, 3)
print("---- nd4 ----")
print(nd4)

np.random.shuffle(nd4)
print("---- shuffle nd4 ----")
print(nd4)

nd5 = np.zeros([3, 3])
print("---- nd5 ----")
print(nd5)

nd6 = np.ones([3, 3])
print("---- nd6 ----")
print(nd6)

nd7 = np.eye(3)
print("---- nd7 ----")
print(nd7)

nd8 = np.diag([1, 2, 3])
print("---- nd8 ----")
print(nd8)
