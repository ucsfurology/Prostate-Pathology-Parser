# -*- coding: utf-8 -*-

import copy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from methods.torch.evaluation import getPredsLabels, getScores


def runModel(net, trainLoader, valLoader, args, printBool=False, cuda=True):
    """
    Arguments:
        - net: is a trained convolutional neural network model
        - trainLoader is a torch data loader
        - valLoader is a torch data loader
        - args is a dictionary of arguments
        - printBool is a flag to print out losses
        - cuda is a flag to use gpu or cpu
    returns:
        - net a trained convolutional neural network model
    """
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])

    for epoch in range(args['epochs']):
        running_loss = 0.0
        net = net.train()

        for i, data in enumerate(trainLoader):
            inputs, labels = data
            labels = labels.cuda()
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0 and printBool:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
        if epoch == 0:
            val_scores = getScores(net, valLoader, cuda=True)
            best = val_scores['f1_micro']
        elif epoch%5 == 0:
            val_scores = getScores(net, valLoader, cuda=True)
            if best > val_scores['f1_micro']:
                return net
    return net
        