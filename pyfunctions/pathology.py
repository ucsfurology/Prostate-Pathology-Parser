# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Tuple

prostate_fields = {'classification': ['TreatmentEffect','TumorType','PrimaryGleason','SecondaryGleason','TertiaryGleason',
                          'SeminalVesicleNone','LymphNodesNone','MarginStatusNone','ExtraprostaticExtension',
                          'PerineuralInfiltration','RbCribriform','BenignMargins'],
                   'token_extraction': ['ProstateWeight', 'TumorVolume', 'TStage', 'NStage', 'MStage']}

def getProstateMapping():
    """
    Return dic mapping of original raw labels back to processed labels
    """
    mapping = {"null": 0, "": 0, '0':1, "o": 1, '1':2, '1a':3, '1b':4, '1c':5, '2': 6,
               '2a': 7, '2b':8, '2c':9, '*s':10, '3':11, '3a': 12, '3b': 13,
               '3c': 14, '4': 15, '4a':16, '5': 17, 't2':18, 't2a':19, 'x': 20,
               'adenoca': 21, 'capsmac':21, 'capdctl': 21, 'sarcoma':22, 'mucinous':21,
               'uccinvas':21}
    return mapping

def getProstateStageInverseMapping():
    """
    Return dic mapping of processed labels back to raw labels
    """
    mapping = { 0:'', '1':2, '1a':3, '1b':4, '1c':5, '2': 6,
               '2a': 7, '2b':8, '2c':9, '*s':10, '3':11, '3a': 12, '3b': 13,
               '3c': 14, '4': 15, '4a':16, '5': 17, 't2':18, 't2a':19, 'x': 20, 'adeno':21, 'sarcoma':22}
    mapping = {v: k for k, v in mapping.items()}
    mapping['0'] = ''
    mapping[0] = ''
    mapping[1] = '0'

    return mapping