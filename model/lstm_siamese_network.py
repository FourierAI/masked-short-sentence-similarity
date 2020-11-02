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
from dataprocess.dataset_loader import DatasetLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SimaeseNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SimaeseNet, self).__init__()

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
        last_left_hidden = h_nx.squeeze()
        last_right_hidden = h_ny.squeeze()

        distance = F.pairwise_distance(last_left_hidden, last_right_hidden)

        output_prediction = torch.exp(-1 * distance)

        return output_prediction


if __name__ == "__main__":

    EPOCH = 10
    BATCH_SIZE = 50
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    NUM_LAYER = 1
    LEARNING_RATE = 0.001

    network = SimaeseNet(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYER)

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset = DatasetLoader('train', EMBEDDING_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)

    for i in range(EPOCH):
        loss_list = []
        for step, batch in enumerate(dataloader):
            left_sent, right_sent, scores = batch

            y_bar = network(left_sent.view(BATCH_SIZE, -1, EMBEDDING_DIM), right_sent.view(BATCH_SIZE, -1, EMBEDDING_DIM))

            loss = criterion(y_bar.float(), scores.float())
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(loss_list)/len(loss_list)

        print('The current {} epoch\'s mean loss:{}'.format(i, mean_loss))

