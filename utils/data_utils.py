import os
import re
import argparse
from configs.config_global import RAW_DIR, EXP_TYPES, RAW_DATA_SUFFIX

def get_exp_names():
    """
    Get the names of the file names in the data directory for each brain region
    """
    exp_names = {}
    for exp_type in EXP_TYPES:
        exp_names[exp_type] = []
        for file_name in os.listdir(RAW_DIR):
            if file_name.endswith(RAW_DATA_SUFFIX) and file_name.startswith(exp_type):
                exp_name = re.sub(RAW_DATA_SUFFIX, '', file_name)
                exp_names[exp_type].append(exp_name)
            
    return exp_names