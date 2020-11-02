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
from torch.autograd import Variable
import numpy as np

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

        sents1_embedding = self.sent2embedding(embedding, sents1_words, word_index)
        sents2_embedding = self.sent2embedding(embedding, sents2_words, word_index)

        for index, score in enumerate(scores):
            self.samples.append(
                (torch.tensor(sents1_embedding[index]), torch.tensor(sents2_embedding[index]), torch.tensor(score)))

    def sent2embedding(self, embedding, sents_words, word_index):
        sents_embedding = []
        for sent in sents_words:
            sent_embedding = []
            for word in sent:
                index = word_index[word]
                index_var = Variable(torch.LongTensor([index]))
                word_embedding = embedding(index_var).squeeze().tolist()
                sent_embedding.append(word_embedding)
            sents_embedding.append(sent_embedding)
        return np.array(sents_embedding)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def collate_fn(self, data):
        left_sent = [sq[0] for sq in data]
        right_sent = [sq[1] for sq in data]
        pair_score = [sq[2] for sq in data]

        left_sent_padding = rnn_utils.pad_sequence(left_sent, batch_first=True, padding_value=0)
        right_sent_padding = rnn_utils.pad_sequence(right_sent, batch_first=True, padding_value=0)

        return left_sent_padding, right_sent_padding, torch.tensor(pair_score)


if __name__ == "__main__":
    dataset = DatasetLoader('train', 100)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)

    for step, batch in enumerate(dataloader):
        left_sent, right_sent, score = batch

        print('score', score)
