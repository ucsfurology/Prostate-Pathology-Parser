# -*- coding: utf-8 -*-

import os
import re
from typing import Dict, List, Tuple

prostate_fields = {'classification': ['TreatmentEffect','TumorType','PrimaryGleason','SecondaryGleason','TertiaryGleason',
                          'SeminalVesicleNone','LymphNodesNone','MarginStatusNone','ExtraprostaticExtension',
                          'PerineuralInfiltration','RbCribriform','BenignMargins'],
                   'token_extraction': ['ProstateWeight', 'TumorVolume', 'TStage', 'NStage', 'MStage']}

NUMERIC_PATTERN = re.compile(r"""
    (?:
        (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
        |
        (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
    )
    """, re.VERBOSE)

get_tnm_stage = lambda x: x


def get_float(txt):
    try:
        return max((float(x) for x in NUMERIC_PATTERN.findall(txt)))
    except:
        return 0

postprocess = {'ProstateWeight': get_float,
               'TumorVolume': get_float,
               'TStage': get_tnm_stage,
               'NStage': get_tnm_stage,
               'MStage': get_tnm_stage}

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

def label_correctness(predictions, field):
    predictions['correct'] = 0
    predictions['final_prediction'] = -1
    
    for j in range(len(predictions['label'])):
        
        # Get label and prediction
        label = predictions['label'][j]
        prediction = postprocess[field](predictions['predicted_token'].iloc[j])
        
        if field in ['TStage', 'NStage', 'MStage']:
            stage_type = field[0].lower()
            stage_encoding = stage_type + str(label)
            
            # Encoding for pathologic stage in-text usually starts with pt, yp, or t
            
            if prediction[0:2] == 'pt' or prediction[0:2] == 'yp' or prediction[0] == 't':
                
                if label == 'nan' or label == 'null':
                    # Case: Stage encoding is nan, expect 0 in predicted token or not available
                    if stage_type not in prediction:
                        # Mark as correct
                        predictions['correct'].iloc[j] = 1
                        predictions['final_prediction'].iloc[j] = predictions['label'][j]
                    elif f"{stage_type}0" in prediction:
                        predictions['correct'].iloc[j] = 1
                        predictions['final_prediction'].iloc[j] = predictions['label'][j]
                    else:
                        predictions['final_prediction'].iloc[j] = prediction
                        
                elif stage_encoding in prediction:
                    # Case: Stage encoding is contained within predicted token, mark as correct
                    predictions['correct'].iloc[j] = 1
                    predictions['final_prediction'].iloc[j] = predictions['label'][j]
                else:
                    predictions['final_prediction'].iloc[j] = prediction
            else:
                if label == 'nan' or label == 'null':
                    predictions['correct'].iloc[j] = 1
                    predictions['final_prediction'].iloc[j] = predictions['label'][j]
                    predictions['y_prob'].iloc[j] = 1 - predictions['y_prob'].iloc[j]
        else:
            if label == prediction:
                # Case label equals prediction, mark as correct
                predictions['correct'].iloc[j] = 1
                predictions['final_prediction'].iloc[j] = predictions['label'][j]
            else:
                predictions['final_prediction'].iloc[j] = prediction
            
    return predictions