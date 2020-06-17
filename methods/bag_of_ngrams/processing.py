# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import sklearn
import random

from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from pyfunctions.general import *
    
"""
Characters to strip from pathology reports
"""
STRIPCHARS = ['\"', ',', ';', 'null', '*', '#', '~', '(', ')' ,"\"", '\'']
SPLITS = ['train','val','test']

def getCounter(data):
    """
    Arguments:
        - data: a list of dictionaries representing patients
    returns:
        - counter: a dictionary of counts of words
    """
    lst = []
    for patient in data:
        doc = patient['clean_document']
        lst = lst + doc.split()
    counter = Counter(lst)
    return counter

def cleanSplit(data, stripObjects, include_dash=False):
    """
    Arguments:
        - data: a list of dictionaries representing patients
        - stripObjects: a string of characters to remove from documents
        - include_dash: a flag to remove dashes or not
    returns:
        - data
    """
    for split in SPLITS:
        data[split] = cleanReports(data[split], stripObjects, include_dash)
            
    return data

def cleanReport(report, chars, include_dash=False):
    """
    Arguments:
        - report: a string
        - chars: characters to remove
        - include_dash: a flag to remove dashes or not
    returns:
        - processed: a processed version of report
    """
    report = report.lower()
    for c in chars:
        report = report.replace(c, ' ')

    """
    Replace / and = characters with spaces preceding and following it
    """
    for c in ['+','/', '=' , ':', '(', ')','<','>']:
        report = report.replace(c, ' '+ c + ' ')
    if include_dash:
        report = report.replace('-', ' - ')

    """
    Remove periods
    """
    processed = []
    for token in report.split():
        token = token.rstrip('.')
        token = token.strip()
        token = token.rstrip('.')
        processed.append(token)
    processed = " ".join(processed)

    return " ".join(processed.split())

def cleanReports(data, stripObjects, include_dash=False):
    """
    Arguments:
        - data: a list of dictionaries representing patients
        - stripObjects: a string of characters to remove
        - include_dash: a flag to remove dashes or not
    returns:
        - data
    """
    report_list = []
    for i, patient in enumerate(data):
        clean_report = cleanReport(patient['document'], stripObjects, include_dash=include_dash)
        data[i]['clean_document'] = clean_report
        report_list.append(clean_report)
    return data

def unkReport(report,counter):
    """
    Arguments:
        - report: a string
        - counter: a dictionary representing counts of vocab
    returns:
        - processed: a report with rare words replaced by <UNK>
    """
    processed = []
    for token in report.split():
        if counter[token] < 2:
            processed.append("<unk>")
        else:
            processed.append(token)
    processed = " ".join(processed)
    return processed  

def unkReports(data, counter):
    """
    Arguments:
        - data: a list of dictionaries representing patients
        - counter: a dictionary representing counts of vocab
    returns:
        - data
    """
    processed_reports = []
    for i, patient in enumerate(data):
        report = patient['clean_document']
        processed = []
        for token in report.split():
            if counter[token] < 2:
                processed.append("<unk>")
            else:
                processed.append(token)
        processed = " ".join(processed)
        patient['clean_document_unked'] = processed
        data[i] = patient
    return data

def getTrainedVectorizer(corpus, N, min_n):
    """
    Arguments:
        - corpus: a list of strings of documents
        - N: max N-gram to use
        - min_n: min N-gram to use
    returns:
        - textVectorizer: sklearn count vectorizer fitted
    """
    textVectorizer = CountVectorizer(stop_words=None,
                                tokenizer=TreebankWordTokenizer().tokenize, 
                                ngram_range = (min_n,N))
    textVectorizer.fit(corpus)
    return textVectorizer
