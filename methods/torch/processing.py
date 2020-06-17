import copy
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from pyfunctions.general import *

def getVocab(data):
    """
    Arguments:
        - data: a list of dictionaries containing attributes of patient data
    returns:
        - vocab: a list of unique vocabulary words
    """
    vocab = []
    for patient in data:
        document = patient['clean_document'].split()
        for word in document:
            if word not in vocab:
                vocab.append(word)
    return vocab

def encodeLabels(data, encoder, field):
    """
    Arguments:
        - data: a list of dictionaries containing attributes of patient data
        - encoder: a trained label encoder mapping raw labels to 0, 1, 2... values
        - field: a data field string
    returns:
        - data
    """
    for i, patient in enumerate(data):
        encoded = encoder.transform([str(patient['labels'][field])])[0]
        if 'encoded_labels' in patient:
            patient['encoded_labels'][field] = encoded
        else:
            patient['encoded_labels'] = {field: encoded}
        data[i] = patient
    return data

def getEncoder(data, field):
    """
    Arguments:
        - data: a list of dictionaries containing attributes of patient data
        - field: a data field string
    returns:
        - encoder: a trained label encoder mapping raw labels to 0, 1, 2... values
    """
    train = []
    for patient in data:
        train.append(str(patient['labels'][field]))
    encoder = LabelEncoder()
    encoder.fit(train)
    return encoder

def reSample(corpus, labels):
    """
    Arguments:
        - corpus: a list of strings reprsenting documents
        - labels: a list of label strings
    returns:
        - corpus_lst: a list of upsampled documents
        - labels_lst: a list of upsampled labels
    """
    corpus_lst = corpus

    labels_lst = copy.deepcopy(labels)
    """
    Get the maximum number a class appears in the labels
    """
    max_occur = getNumMaxOccurrences(labels)

    """
    Get list containing each class index set (indices where a class shows up)
    """
    classIndices = getClassIndices(np.array(labels))

    """
    Loop through each class index set
    """
    for indices in classIndices:
        """
        Get subset that matches current class
        """
        subLabels = np.array(labels)[indices]
        subCorpus = [data for i, data in enumerate(corpus) if i in indices ]

        """
        Upsample the class so that the number of instances matches max_occur
        """
        n = len(subLabels)
        upsampled_indices = np.random.choice(range(n), size=max_occur - n, replace=True)
        if len(upsampled_indices) > 0:
            upsampled_labels = subLabels[upsampled_indices]
            upsampled_corpus = [subCorpus[ind] for ind in upsampled_indices]

            """
            Add upsampled data to report and labels lists
            """
            corpus_lst = corpus_lst + upsampled_corpus
            labels_lst = labels_lst + upsampled_labels.tolist()
    return corpus_lst, labels_lst


def getTorchLoader(corpus, labels, args, shuffle):
    """
    Arguments:
        - corpus: a list of strings reprsenting documents
        - labels: a list of label strings
        - args: a dic of arguments
        - shuffle: a flag to shuffle the data or not
    returns:
        - loader: a pytorch data loader
    """
    features = torch.zeros(len(corpus), args['maxDocLength'], dtype = torch.long)

    for i, doc in enumerate(corpus):
        doc = doc.split()
        if len(doc) > args['maxDocLength']:
            doc = doc[0:args['maxDocLength']]

        docVec = torch.zeros(args['maxDocLength'], dtype = torch.long)

        j = args['maxDocLength'] - 1
        for word in list(reversed(doc)):
            docVec[j] = args['word2idx'][word]
            j-=1

        features[i,:] = docVec

    targets = torch.LongTensor(labels)
    dataset = TensorDataset(features, targets)

    loader = DataLoader(dataset, batch_size= args['batchSize'], shuffle=shuffle, num_workers=0)
    return loader