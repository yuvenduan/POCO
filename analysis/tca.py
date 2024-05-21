import torch
import torch.nn as nn
import os
import tensorly as tl
import matplotlib.pyplot as plt

from configs.configs import NeuralPredictionConfig
from datasets.zebrafish import Zebrafish
from configs.config_global import DEVICE
from models import SimpleRNN
from tensorly.decomposition import parafac
from utils.train_utils import model_init

@torch.no_grad()
def tca(config: NeuralPredictionConfig):
    # get the data
    dataset = Zebrafish(config, phase='train')
    neural_data = dataset.neural_data

    # init model
    model_path = os.path.join(config.save_path, 'net_best.pth')
    state_dict = torch.load(model_path, map_location=DEVICE)
    model: SimpleRNN = model_init(config, datum_size=dataset.datum_size)
    model.load_state_dict(state_dict)
    model.eval()

    # do analysis on each fish
    for fish_id, activity in enumerate(neural_data):

        bsz = config.batch_size
        current_list = []
        activity = activity[:, : -1]

        for i in range(0, activity.shape[1], bsz):
            batch_activity = activity[:, i: i + bsz].T # T * N
            batch_activity = torch.tensor(batch_activity).float().to(DEVICE)
            embedding = model.in_proj[fish_id](batch_activity)
            linear: nn.Linear = model.rnn[0]

            for idx in range(embedding.shape[0]):
                current_list.append(linear.weight * embedding[idx])

        # get_tca
        currents = torch.stack(current_list).cpu().numpy()
        print("Current Shape", currents.shape)
        weights, factors = parafac(currents, rank=5, normalize_factors=True, tol=1e-10) # 3 * (shape) * 5
        print(weights)

        os.makedirs(os.path.join(config.save_path, 'tca'), exist_ok=True)
        for idx in range(5):
            plt.plot(factors[0][:, idx])
            plt.title(f'Factor {idx}')
            plt.savefig(os.path.join(config.save_path, 'tca', f'fish_{fish_id}_factor_{idx}.png'))
            plt.close()
