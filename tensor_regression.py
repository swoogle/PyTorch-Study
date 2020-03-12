#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

from __future__ import print_function
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(100)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = 3 * x ^ 2 + 2, 加上一些噪声数据
y = 3 * x.pow(2) + 2 + 0.2 * torch.rand(x.size())

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# 随机初始化参数
w = torch.rand(1, 1, dtype=torch.float, requires_grad=True)
b = torch.zeros(1, 1, dtype=torch.float, requires_grad=True)
# 设定学习率
lr = 0.001
for i in range(1000):
    # 前向传播
    y_pred = x.pow(2).mm(w) + b
    # 定义损失函数 
    loss = torch.sum(0.5 * (y_pred - y) ** 2)
    # 自动计算梯度，梯度存放在grad属性
    loss.backward()
    # 更新参数，使上下文环境中切断自动求导计算
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    # 梯度清零
    w.grad.zero_()
    b.grad.zero_()

plt.plot(x.numpy(), y_pred.detach().numpy(), 'r-', label='predict')
plt.scatter(x.numpy(), y.numpy(), color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()
print("w = {}, b = {}".format(w, b))
