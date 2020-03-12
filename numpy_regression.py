#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1)
# y = 3 * x ^ 2 + 2, 加上一些噪声数据
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, 1)

# plt.scatter(x, y)
# plt.show()

# 随机初始化参数
w = np.random.rand(1, 1)
b = np.random.rand(1, 1)
# 设定学习率
lr = 0.001
for i in range(1000):
    # 前向传播
    y_pred = w * np.power(x, 2) + b
    # 定义损失函数 
    loss = 0.5 * np.sum((y_pred - y) ** 2)
    # 计算梯度
    grad_w = np.sum((y_pred - y) * np.power(x, 2))
    grad_b = np.sum((y_pred - y))
    # 采用梯度下降法，使loss最小
    w -= lr * grad_w
    b -= lr * grad_b

plt.plot(x, y_pred, 'r-', label='predict')
plt.scatter(x, y, color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()
print("w = {}, b = {}".format(w, b))
