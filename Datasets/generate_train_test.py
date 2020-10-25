#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: generate_train_test.py
# @time: 2020-10-25 22:37
# @desc:

import random

if __name__ == "__main__":

    lines = []
    with open("datasets_all.txt") as file:
        for line in file:
            lines.append(line)

    random.shuffle(lines)

    with open('train.txt', 'w') as file:
        file.write(''.join(lines[:90000]))

    with open('test.txt', 'w') as file:
        file.write(''.join(lines[90000:]))