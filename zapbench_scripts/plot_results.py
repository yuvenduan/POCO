from zapbench import constants
"""Check the model performance in Zapbench"""

from connectomics.common import ts_utils
import pandas as pd
import numpy as np
import os
import torch

# Load dataframe with results reported in the manuscript.
df = pd.DataFrame(
    ts_utils.load_json(f'gs://zapbench-release/dataframes/20250131/combined.json'))
print(df.head())

pred_length = 32
avg_mae = {}
sem_mae = {}

baselines = ['unet', 'linear', 'tsmixer']

for context in [4, 256]:
  for method in baselines:

    mae_list = []
    for condition_name in constants.CONDITION_NAMES:
      mae_df = df.query(
          f'method == "{method}" and context == {context} and condition == "{condition_name}"'
      ).sort_values('steps_ahead')

      mae = mae_df['MAE'].to_numpy()
      mae = mae.reshape(pred_length, -1)
      mae_list.append(mae)

    mae_list = np.stack(mae_list, axis=0) # (conditions, pred_length, num_seeds)
    avg_mae[f'{method}_{context}'] = mae_list[constants.CONDITIONS_TRAIN, :].mean(axis=(0, 2))
    sem_mae[f'{method}_{context}'] = mae_list[constants.CONDITIONS_TRAIN, :].mean(axis=0).std(axis=1) / np.sqrt(mae_list.shape[-1] - 1)
    print(method, context, avg_mae[f'{method}_{context}'][[0, 1, 3, 7, 15, 31]], avg_mae[f'{method}_{context}'].mean())

load_dir = "experiments/poco_ctx{}_dctx0_lr0.0003_lossL1Loss_cf{}_wd0.0001_drop0.0_cond1024_seed{}"

for context in [4, 256]:
  mae_list = []
  cf = 4 if context == 4 else 64

  for seed in range(3):
    load_path = load_dir.format(context, cf, seed)
    load_path = os.path.join(load_path, "mae.npy")
    mae, poco_avg_mae = torch.load(load_path, weights_only=False)

    mae_list.append(poco_avg_mae)

  avg_mae[f'poco_{context}'] = np.stack(mae_list, axis=0).mean(axis=0)
  sem_mae[f'poco_{context}'] = np.stack(mae_list, axis=0).std(axis=0) / np.sqrt(len(mae_list) - 1)

# plot the results: step ahead vs MAE

import matplotlib.pyplot as plt
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

BASE_MODEL_COLORS = {
  'linear': '#2C4875', 'poco': '#E67E22', 'unet': '#9B3D3D', 'tsmixer': '#C9A66B',
}

def plot_mae(avg_mae, sem_mae, context):
  # compare the average MAE for each method
  fig, ax = plt.subplots(figsize=(4, 3))
  ax.set_title(f"MAE for C = {context}")
  ax.set_xlabel("Steps ahead")
  ax.set_ylabel("MAE")
  
  for method in ['linear', 'unet', 'tsmixer', 'poco']:
    ax.plot(avg_mae[f'{method}_{context}'], label=method, color=BASE_MODEL_COLORS[method])
    ax.fill_between(
        np.arange(pred_length),
        avg_mae[f'{method}_{context}'] - sem_mae[f'{method}_{context}'],
        avg_mae[f'{method}_{context}'] + sem_mae[f'{method}_{context}'],
        alpha=0.2, color=BASE_MODEL_COLORS[method],
    )

  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join(fig_dir, f"mae_{context}.pdf"))

plot_mae(avg_mae, sem_mae, 4)
plot_mae(avg_mae, sem_mae, 256)