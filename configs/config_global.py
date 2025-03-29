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

ZEBRAFISH_RAW_DIR = os.path.join(DATA_DIR, 'raw_zebrafish')
ZEBRAFISH_AHRENS_RAW_DIR = os.path.join(DATA_DIR, 'raw_zebrafish_ahrens')
EXP_TYPES = ['control', 'shocked', 'reshocked', 'ketamine']
ZEBRAFISH_STIM_RAW_DIR = os.path.join(DATA_DIR, 'raw_zebrafish_stim')
ZEBRAFISH_JAIN_RAW_DIR = os.path.join(DATA_DIR, 'raw_zebrafish_jain')
CELEGANS_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_zimmer')
CELEGANS_FLAVELL_RAW_DIR = os.path.join(DATA_DIR, 'raw_celegans_flavell')
MICE_RAW_DIR = os.path.join(DATA_DIR, 'raw_mice')
MICE_BRAIN_AREAS = ['PPC', 'RSP', 'V1', 'M2']

N_ZEBRAFISH_SESSIONS = 19
N_ZEBRAFISH_AHNRENS_SESSIONS = 15
N_CELEGANS_SESSIONS = 5
N_CELEGANS_FLAVELL_SESSIONS = 40
N_MICE_SESSIONS = 12

RAW_DATA_SUFFIX = '_CNMF_compressed.h5'
ZEBRAFISH_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zebrafish')
ZEBRAFISH_AHRENS_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zebrafish_ahrens')
ZEBRAFISH_STIM_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zebrafish_stim')
CELEGANS_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans_zimmer')
CELEGANS_FLAVELL_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_celegans_flavell')
ZEBRAFISH_JAIN_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_zebrafish_jain')
MICE_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_mice')
SIM_DIR = osp.join(DATA_DIR, 'simulations')

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MAP_LOC = "cuda:0" if USE_CUDA else torch.device('cpu')
LOG_LEVEL = logging.INFO
BASE_MODEL_COLORS = {
    'Linear': '#2C4875', 'DLinear': '#5A7684', 'POYO': '#edae49', 'POCO': '#E67E22', # '#FF7F0E', #
    'Latent_PLRNN': '#9B3D3D', 'AR_Transformer': '#005F73', 
    'PaiFilter': '#A0C4FF', 'TexFilter': '#68A4D3',
    'TCN': '#E99675', 'MLP': '#4E5D6C', 'TSMixer': '#C9A66B',
}
MODEL_COLORS = {}

for model in BASE_MODEL_COLORS:
    MODEL_COLORS['All_' + model] = BASE_MODEL_COLORS[model] + '60'
    MODEL_COLORS['MS_' + model] = BASE_MODEL_COLORS[model]
    MODEL_COLORS[model] = BASE_MODEL_COLORS[model] + '85'