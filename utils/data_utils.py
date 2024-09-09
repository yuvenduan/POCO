import os
import re
import argparse
from configs.config_global import RAW_DIR, EXP_TYPES, RAW_DATA_SUFFIX, VISUAL_PROCESSED_DIR

def get_exp_names():
    """
    Get the experiment names for each experiment type (for spontaneous fish activity)
    """
    exp_names = {}
    for exp_type in EXP_TYPES:
        exp_names[exp_type] = []
        for file_name in os.listdir(RAW_DIR):
            if file_name.endswith(RAW_DATA_SUFFIX) and file_name.startswith(exp_type):
                exp_name = re.sub(RAW_DATA_SUFFIX, '', file_name)
                exp_names[exp_type].append(exp_name)
            
    return exp_names

def get_subject_ids():
    """
    Get the subject ids (for fish data with visual stimuli)
    """
    subject_ids = []
    for file_name in os.listdir(VISUAL_PROCESSED_DIR):
        if file_name.endswith('.npz') and file_name.startswith('subject'):
            subject_id = re.sub('.npz', '', file_name)
            subject_id = re.sub('subject_', '', subject_id)
            subject_ids.append(int(subject_id))

    return subject_ids

def get_stim_exp_names():
    exp_names = {
        'control': ['c02'],
        'stim': ['e01', 'e02', 'e03']
    }
    return exp_names