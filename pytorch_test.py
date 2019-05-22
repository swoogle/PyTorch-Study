#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

from __future__ import print_function
import torch


x = torch.rand(5, 3)
print(x)
print('cuda = %s' % str(torch.cuda.is_available()))