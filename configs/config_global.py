import os
import os.path as osp
import logging
import torch

NP_SEED = 233
TCH_SEED = 3407

# Directory paths
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
FIG_DIR = osp.join(ROOT_DIR, 'figures')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

RAW_DIR = os.path.join(DATA_DIR, 'raw')
EXP_TYPES = ['control', 'shocked', 'reshocked', 'ketamine']
STIM_RAW_DIR = os.path.join(DATA_DIR, 'raw_stim')
VISUAL_RAW_DIR = os.path.join(DATA_DIR, 'raw_visual')
CELEGANS_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_zimmer')
CELEGANS_FLAVELL_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_flavell')
MICE_RAW_DIR = os.path.join(DATA_DIR, 'raw_mice')
MICE_BRAIN_AREAS = ['PPC', 'RSP', 'V1', 'M2']

N_ZEBRAFISH_SESSIONS = 19
N_CELEGANS_SESSIONS = 5
N_CELEGANS_FLAVELL_SESSIONS = 40
N_MICE_SESSIONS = 12

RAW_DATA_SUFFIX = '_CNMF_compressed.h5'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zscored')
VISUAL_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_visual_zscored')
STIM_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_stim_zscored')
CELEGANS_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans')
CELEGANS_FLAVELL_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans_flavell')
MICE_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_mice')
SIM_DIR = osp.join(DATA_DIR, 'simulations')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device('cpu')
LOG_LEVEL = logging.INFO
BASE_MODEL_COLORS = {'Linear': '#003D5B', 'POYO': '#EDAE49', 'Predict-POYO': '#EDAE49', 'Latent_PLRNN': '#D1495B', 'AR_Transformer': '#00798C', 'TCN': '#ADD8E6', }
MODEL_COLORS = {}

for model in BASE_MODEL_COLORS:
    MODEL_COLORS['All_' + model] = BASE_MODEL_COLORS[model]
    MODEL_COLORS['MS_' + model] = BASE_MODEL_COLORS[model] + '85'
    MODEL_COLORS[model] = BASE_MODEL_COLORS[model] + '60'