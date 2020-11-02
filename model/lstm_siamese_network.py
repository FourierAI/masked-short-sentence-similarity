#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: lstm_siamese_network.py
# @time: 2020-10-26 20:49
# @desc:


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataprocess.dataset_loader import DatasetLoader


class SiameseNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SiameseNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Default Bert dimension 768
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # e.g. (batch, time_step, input_size)
        )

    def forward(self, x, y):
        x_out, (h_nx, h_cx) = self.rnn(x, None)
        y_out, (h_ny, h_cy) = self.rnn(y, None)

        BATCH_SIZE = x.shape[0]
        last_left_hidden = h_nx.view(BATCH_SIZE, self.hidden_dim)
        last_right_hidden = h_ny.view(BATCH_SIZE, self.hidden_dim)


        distance = F.pairwise_distance(last_left_hidden, last_right_hidden)

        output_prediction = torch.exp(-1 * distance)

        return output_prediction


if __name__ == "__main__":

    EPOCH = 1
    BATCH_SIZE = 100
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    NUM_LAYER = 1
    LEARNING_RATE = 0.01

    network = SiameseNet(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYER)
    NETWORK_PATH = 'siamese_lstm.pkt'

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset_train = DatasetLoader('train', EMBEDDING_DIM)
    dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            collate_fn=dataset_train.collate_fn)

    for i in range(EPOCH):
        loss_list = []
        for step, batch in enumerate(dataloader):
            left_sent, right_sent, scores = batch

            y_bar = network(left_sent.view(BATCH_SIZE, -1, EMBEDDING_DIM),
                            right_sent.view(BATCH_SIZE, -1, EMBEDDING_DIM))

            loss = criterion(y_bar.float(), scores.float())
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(loss_list) / len(loss_list)

        print('The current {} epoch\'s mean loss:{}'.format(i, mean_loss))

    torch.save(network.state_dict(), NETWORK_PATH)

    inference_network = SiameseNet(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYER)
    inference_network.load_state_dict(torch.load(NETWORK_PATH))

    dataset_test = DatasetLoader('test', EMBEDDING_DIM)
    Y_PREDICTED = []
    scores = []
    for sent1, sent2, score in dataset_test:
        scores.append(score.item())
        y_bar = inference_network(sent1.view(1, -1, EMBEDDING_DIM), sent2.view(1, -1, EMBEDDING_DIM))
        if y_bar > 0.4:
            Y_PREDICTED.append(1)
        else:
            Y_PREDICTED.append(0)

    # compute accuracy and F1
    accuracy = accuracy_score(scores, Y_PREDICTED)
    F1 = f1_score(scores, Y_PREDICTED)

    print('model performance, accuracy:{},F1:{}'.format(accuracy, F1))
