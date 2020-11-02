#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/11/2 23:25
# @Author : Zhipeng Ye
# @File : test.py
# @desc :
import torch.nn as nn
import torch

sm = nn.Softmax(dim=0)
out = sm(torch.tensor([[0.5], [0.6]]))
print(out)
