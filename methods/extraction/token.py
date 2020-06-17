# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Sequence

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from pyfunctions.general import *
from methods.extraction.general import getDocumentTokenTypes
    

def populateLabels(tuples, labels, args, stage):
    """
    Given a list of tuples (word, context, label), give a 1 value for 
    the label if the word matches the actual token label
    """
    for i, tokenTuple in enumerate(tuples):
        word, word_type, context, label = tokenTuple
        if not stage:
            if str(labels) == word:
                """
                Case: token label directly matches word
                """
                tuples[i] = (word, word_type, context, 1)
            else:
                """
                Case: token label does not match word
                """
                tuples[i] = (word, word_type, context, 0)
        else:
            labels = str(labels)
            if labels in args['stage_mapping'].values() and labels in word.lower() and 'pt' in word.lower() and 'p' == word.lower()[0] and 't' == word.lower()[1]:
                """
                Case: token label is in stage encoding
                """
                tuples[i] = (word, word_type, context, 1)
            elif str(labels) == '0' and 'pt' in word.lower() and 'p' == word.lower()[0] and 't' == word.lower()[1]:
                tuples[i] = (word, word_type, context, 1)
            else:
                """
                Case: token label does not match word
                """
                tuples[i] = (word, word_type, context, 0)

    return tuples   

def getTuples(document, wordTypes, context_size=1):
    """
    For a document, return a list of tuples (word, context, type, label)
    Initialize the label to be 0 for now
    """
    k = context_size
    lst = []
    n = len(document)
    for i, token in enumerate(document):
        if i >= k and i < n-k-1:
            """
            if token is in body of document
            """
            word = token
            context = []
            for j in range(i-k, i+k+1):
                context.append(document[j])
            lst.append((word, wordTypes[i], context,0))
        elif i < k:
            """
            if token is one of first k tokens in document
            """
            word = token
            context = []
            for j in range(0, 2*k+1 ):
                context.append(document[j])
            lst.append((word,wordTypes[i], context,0))
        else:
            """
            if token is one of last k tokens in document
            """
            word = token
            context = []
            for j in range(n-2*k-1, n ):
                context.append(document[j])
            lst.append((word,wordTypes[i], context, 0))
    return lst

def tuples2Y(data):
    """
    Given the data as a list of tuples containing the word, word type,
    context, and label; return the labels as a numpy array
    """
    y = []
    for j, tuples in enumerate(data):
        word, word_type, context, label = tuples
        y.append(label)
    return np.array(y)


def _exclude_rare(context, counter):
    """blanks out rare ngrams in-place"""
    for i, context_word in enumerate(context):
        """
        Unk word if needed
        """
        if counter[context_word] < 2:
            context[i] = "<unk>"

def tuples2X(tuples, vectorizer, counter: [Dict, None] = None):
    """
    Given the data as a list of tuples containing the word, word type,
    context, and label; return the X features scipy sparse matrix
    """
    X = []
    for i, tokenTuple in enumerate(tuples):
        word, word_type, context, label = tokenTuple
        if counter is not None:
             _exclude_rare(context, counter) # in-place
        context = " ".join(context)
        """
        Get bag of Ngrams representation of the context
        """

        word_vec = vectorizer.transform([word])
        vec = vectorizer.transform([context])
        """
        Set type to be 0 if token is a word and 1 if numeric value
        """
        type_vec = np.zeros(3)
        type_vec[word_type-1] = 1
        """
        Concatenate bag of Ngrams vector and type value
        """
        vec = hstack([vec + word_vec*5, csr_matrix(type_vec)])
        vec = csr_matrix(vec)
        X.append(vec)
    return vstack(X)

def getX(report, args):
    """
    Given a report, convert it to a sprase matrix and return it
    """
    report = report.split()
    types = getDocumentTokenTypes(report)
    tuples = getTuples(report, types, context_size=args['k'])

    """
    Convert report to string
    """
    report = elements2str(report)
    X = tuples2X(tuples, args['vectorizer'], args['counter'])
    return X

def getY(report, label, args, stage):
    """
    Given a report and a label, extract the y array
    (binary labels for all tokens in the report)
    """
    report = report.split()
    types = getDocumentTokenTypes(report)
    tuples = getTuples(report, types, context_size=args['k'])

    if label == 0 and not stage:
        """
        If label = 0, then it means the label is NULL and we 
        don't want to extract anything
        """
        label = -1

    """
    Convert report to string
    """
    report = elements2str(report)
    tuples = populateLabels(tuples, label, args, stage)
    y = tuples2Y(tuples)
    return y
