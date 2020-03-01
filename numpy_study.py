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

nd9 = np.arange(10)
print("---- nd9 ----")
print(nd9)

nd10 = np.arange(1, 4, 0.5)
print("---- nd10 ----")
print(nd10)

nd11 = np.arange(9, -1, -1)
print("---- nd11 ----")
print(nd11)

nd12 = np.linspace(0.1, 1, 10)
print("---- nd12 ----")
print(nd12)

nd13 = np.arange(10)
print("---- nd13 ----")
print(nd13)
# 获取指定位置数据
print(nd13[3])
# 获取一段数据
print(nd13[3:6])
# 获取固定间隔数据
print(nd13[1:6:2])
# 倒序取数
print(nd13[::-2])

nd14 = np.arange(25).reshape([5, 5])
print("---- nd14 ----")
print(nd14)
#截取一个多维数组中，数值在一个值域的数据
print(nd14[(nd14 > 3) & (nd14 < 10)])
# 获取多维数组指定的行
print(nd14[[1, 2]])
print(nd14[1:3, :])
# 获取多维数组指定的列
print(nd14[:, [1, 2]])
print(nd14[:, 1:3])
#截取一个多维数组中一个区域的数据
print(nd14[1:3, 1:3])

nd15 = np.arange(1, 25, dtype=float)
print("---- nd15 ----")
print(nd15)
# 随机可重复抽取数据
c1 = np.random.choice(nd15, size=(3, 4))
print(c1)
# 随机不重复抽取数据
c2 = np.random.choice(nd15, size=(3, 4), replace=False)
print(c2)
# 随机安指定概率抽取数据
c3 = np.random.choice(nd15, size=(3, 4), p= nd15 /np.sum(nd15))
print(c3)

A = np.array([[1, 2], [-1, 4]])
B = np.array([[2, 0], [3, 4]])
nd16 = np.multiply(A, B)
print("---- nd16 ----")
print(nd16)

def softmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

nd17 = np.random.rand(2, 3)
print("---- nd17 ----")
print(nd17)
print("softmoid(nd17) = \n", softmoid(nd17))
print("relu(nd17) = \n", relu(nd17))
print("softmax(nd17) = \n", softmax(nd17))

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6, 7], [8, 9, 10]])
nd18 = np.dot(A, B)
print("---- nd18 ----")
print(nd18)
