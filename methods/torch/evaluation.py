import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.autograd import Variable


def getPredsLabels(model, loader, probs=False, cuda=False):
    """
    Arguments:
        - model: is a trained convolutional neural network model
        - loader is a torch data loader
        - probs is a flag to optionally return model probabilities
        - cuda is a flag to use gpu or cpu
    returns:
        - Predictions of model
        - Labels of model
    """
    testPreds = []
    testLabels = []
    testProbs = []
    
    for batch, labels in loader:
        if cuda:
            batch = batch.cuda()
            labels = labels.cuda()
        else:
            batch = batch.cpu()
            labels = labels.cpu()
        output = model.forward(Variable(batch))
        values, indices = torch.max(output, 1)
        testPreds = testPreds + np.array(indices.data.cpu()).tolist()
        testLabels = testLabels + np.array(labels.data.cpu()).tolist()
        testProbs += np.array(output.data.cpu()).tolist()
    if probs:
        return np.array(testPreds), np.array(testLabels), np.array(testProbs)
    return np.array(testPreds), np.array(testLabels)

def getScores(model, loader, cuda=False):
    """
    Arguments:
        - model: is a trained convolutional neural network model
        - loader is a torch data loader
        - cuda is a flag to use gpu or cpu
    returns:
        - scores a dictionary containing f1, precision, and recall by various weighting methods
    """
    preds, labels = getPredsLabels(model, loader, probs=False, cuda=cuda)

    scores = {}
    scores['f1_weighted'] = f1_score(labels,preds, average='weighted', pos_label=None)
    scores['f1_macro'] = f1_score(labels,preds, average='macro', pos_label=None)
    scores['f1_micro'] = f1_score(labels,preds, average='micro', pos_label=None)
    scores['f1_by_class'] = f1_score(labels,preds, average=None, pos_label= None).tolist()

    scores['precision_weighted'] = precision_score(labels, preds, average='weighted', pos_label=None)
    scores['precision_macro'] = precision_score(labels, preds, average='macro', pos_label=None)
    scores['precision_micro'] = precision_score(labels, preds, average='micro', pos_label=None)
    scores['precision_by_class'] = precision_score(labels, preds, average=None, pos_label=None).tolist()

    scores['recall_weighted'] = recall_score(labels, preds, average='weighted', pos_label=None)
    scores['recall_macro'] = recall_score(labels, preds, average='macro', pos_label=None)
    scores['recall_micro'] = recall_score(labels, preds, average='micro', pos_label=None)
    scores['recall_by_class'] = recall_score(labels, preds, average=None, pos_label=None).tolist()
    return scores