#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: dataset_loader.py
# @time: 2020-10-25 22:08
# @desc:

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from dataprocess.word_embedding import *


class DatasetLoader(Dataset):

    def __init__(self, dataset_type, embedding_dim):

        self.samples = []

        word_index, embedding = get_word_embedding(embedding_dim, '../Datasets/datasets_all.txt')
        self.word_index = word_index
        self.embedding = embedding

        train_path = '../Datasets/train.txt'
        test_path = '../Datasets/test.txt'

        if dataset_type == "train":
            sents1, sents2, scores = list_sentpair_score(train_path)

        else:
            sents1, sents2, scores = list_sentpair_score(test_path)

        sents1_words = cut_word(sents1)
        sents2_words = cut_word(sents2)

