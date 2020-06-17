# -*- coding: utf-8 -*-

import json
import numpy as np
import os
    
def createDirectory(path):
    """
    Create a directory at a given path if it does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)

def lst2str(lst):
    """
    Join the elements of a list to a string and return the string
    """
    s = " ".join(lst)
    return s

def elements2str(lst):
    """
    Maps sublists of a list to a string
    """
    processed = []
    for element in lst:
        processed.append(lst2str(element))
    return processed

def saveJson(path, obj):
    """
    Save a python object at a given path 
    """
    with open(path, 'w') as outfile:
        json.dump(obj, outfile)

def readJson(path):
    """
    * Load a json file at a given path and return the python object
    """
    with open(path, 'r') as fp:
        data = json.load(fp)
        return data

def getUniqueColValues(labels):
    """
    Given a list of lists, return a list of lists that contain the
    unique values of each list element
    """
    columns = [ [] for i in range(len(labels[0]))]

    for patient in labels:
        for x, label in enumerate(patient):
            if label not in columns[x]:
                columns[x].append(label)
    return columns

def getNumMaxOccurrences(lst):
    """
    Given a list, return the number of times the mode appears
    """
    max_occur = max(lst,key=lst.count)
    return lst.count(max_occur)

def getClassIndices(y):
    """
    Given a list of values, find all the unique values that occur
    and return a list containing lists that contain the indices
    that containo each unique value
    """
    vals = np.unique(y)
    lsts = []

    for val in vals:
        lsts.append(np.where(y == val)[0].tolist())
    return lsts

def combineListElements(lst1, lst2, combineChar='_'):
    """
    Given two lists of strings, return a single list containing
    the concatenation of the strings in each list
    """
    combined = []
    for i in range(len(lst1)):
        combined.append( lst1[i] + combineChar + lst2[i])
    return combined

def listFilesType(path, fileType):
    """
    Given a path, list all files that are of type fileType
    """
    lst = []
    for f in os.listdir(path):
        if f.endswith(fileType):
            lst.append(f)
    return lst
    
def hasNumeric(word):
    """
    Check whether a word has a digit
    """
    for i in range(10):
        if str(i) in word:
            return True
    return False

def hasLetter(word):
    """
    Check whether a word has a letter of alphabet
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for letter in letters:
        if letter in word:
            return True
    return False

def extractListFromDic(lst, key, additional_key=None):
    """
    Extract an attribute of a dic from a list of dics and return as a list
    """
    extracted = []
    
    for i in range(len(lst)):
        if additional_key == None:
            extracted.append(lst[i][key])
        else:
            extracted.append(lst[i][key][additional_key])
    
    return extracted