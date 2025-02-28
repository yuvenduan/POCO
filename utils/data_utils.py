import os
import re
from configs.config_global import ZEBRAFISH_RAW_DIR, EXP_TYPES, RAW_DATA_SUFFIX, MICE_RAW_DIR

def get_exp_names():
    """
    Get the experiment names for each experiment type (for spontaneous zebrafish activity)
    """
    exp_names = {}
    for exp_type in EXP_TYPES:
        exp_names[exp_type] = []
        for file_name in os.listdir(ZEBRAFISH_RAW_DIR):
            if file_name.endswith(RAW_DATA_SUFFIX) and file_name.startswith(exp_type):
                exp_name = re.sub(RAW_DATA_SUFFIX, '', file_name)
                exp_names[exp_type].append(exp_name)
            
    return exp_names

def get_stim_exp_names():
    exp_names = {
        'control': [f'c0{x}' for x in range(4)],
    }
    return exp_names

def get_mice_sessions():
    all_sessions = {}
    # get all directories in the mice raw data directory
    for dir_name in os.listdir(MICE_RAW_DIR):
        mice_dir = os.path.join(MICE_RAW_DIR, dir_name)
        session_list = []
        if os.path.isdir(mice_dir):
            for session_name in os.listdir(mice_dir):
                session_dir = os.path.join(mice_dir, session_name)
                if os.path.isdir(session_dir):
                    session_list.append(session_name)
            all_sessions[dir_name] = session_list

    return all_sessions