#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import numpy as np

# 生成10000个 2X3矩阵，第一个参数为样本数，后两个参数是矩阵形状
data_train = np.random.rand(10000, 2, 3)
# 打乱数据
np.random.shuffle(data_train)
# 定义批量大小
batch_size = 100
# 批处理
for i in range(0, len(data_train), batch_size):
    x_batch_sum = np.sum(data_train[i:i + batch_size])
    print("第{}批次，该批次数据之和：{}".format(i, x_batch_sum))