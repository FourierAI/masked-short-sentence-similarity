#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: word_embedding.py
# @time: 2020-10-22 23:48
# @desc:
import jieba
import torch.nn as nn
import pickle
import torch
import os


def cut_word(sents):
    sents_words = []
    for sent in sents:
        words = jieba.lcut(sent)
        sents_words.append(words)
    return sents_words


def generate_word_index(sents):
    word_index = {}
    count = 0
    for sent in sents:
        for word in sent:
            if word not in word_index:
                word_index[word] = count
                count += 1

    return word_index


def get_word_embedding(embedding_dim, file_path):
    embedding_path = '../Datasets/word_embedding.pt'
    word_index_path = '../Datasets/word_index.pt'

    if os.path.exists(embedding_path) and os.path.exists(word_index_path):
        embedding = torch.load(embedding_path)
        with open(word_index_path, 'rb') as file:
            word_index = pickle.load(file)
    else:
        sents1, sents2, scores = list_sentpair_score(file_path)

        sents1_words = cut_word(sents1)
        sents2_words = cut_word(sents2)

        sents_words = []
        sents_words.extend(sents1_words)
        sents_words.extend(sents2_words)

        word_index = generate_word_index(sents_words)
        word_size = len(word_index)

        embedding = nn.Embedding(word_size, embedding_dim)

        torch.save(embedding, embedding_path)
        with open(word_index_path, 'wb') as file:
            pickle.dump(word_index, file)

    return word_index, embedding


def list_sentpair_score(file_path):
    sents1 = []
    sents2 = []
    scores = []
    with open(file_path) as file:
        for line in file:
            contents = line.split('\t')
            sent1, sent2, score = contents
            sents1.append(sent1)
            sents2.append(sent2)
            scores.append(int(score.strip()))
    return sents1, sents2, scores


if __name__ == "__main__":
    word_index, embedding = get_word_embedding(100, '../Datasets/datasets_all.txt')
