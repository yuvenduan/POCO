# -*- coding: utf-8 -*-
"""zapbench.ipynb
Adapted from https://github.com/google-research/zapbench/blob/main/colabs/train_and_evaluate.ipynb
"""

from zapbench import constants
from zapbench import data_utils
from zapbench.ts_forecasting import data_source

"""We can also create a data source that combines data from all training conditions (should take about a minute to prefetch):"""

def get_source(num_timesteps_context, split='train'):

  sources = []

  # Iterate over all training conditions (excludes 'taxis'), and create
  # data sources.
  for condition_id in constants.CONDITIONS_TRAIN:
    config = data_source.TensorStoreTimeSeriesConfig(
        input_spec=data_utils.adjust_spec_for_condition_and_split(
            condition=condition_id,
            split=split,
            spec=data_utils.get_spec('240930_traces'),
            num_timesteps_context=num_timesteps_context),
        timesteps_input=num_timesteps_context,
        timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
    )
    sources.append(data_source.TensorStoreTimeSeries(config, prefetch=True))

  # Concatenate into a single source.
  source = data_source.ConcatenatedTensorStoreTimeSeries(*sources)

  return source

"""Next, we set up an index sampler and construct a data loader with `grain`:"""

import grain.python as grain

def get_data_loader(source, batch_size=4, num_epochs=1, seed=42):

  shuffle = True

  index_sampler = grain.IndexSampler(
      num_records=len(source),
      num_epochs=num_epochs,
      shard_options=grain.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=True),
      shuffle=shuffle,
      seed=seed
  )

  data_loader = grain.DataLoader(
      data_source=source,
      sampler=index_sampler,
      operations=[
          grain.Batch(
              batch_size=batch_size, drop_remainder=True)
      ],
      worker_count=0
  )

  return data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from standalone_poco import POCO, NeuralPredictionConfig

"""Define a simple linear model"""

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs: NeuralPredictionConfig, input_size):
        super().__init__()
        self.task_name = "short_term_forecast"
        self.seq_len = configs.seq_length - configs.pred_length
        self.pred_len = configs.pred_length

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = input_size
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.permute(1,0,2)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x.permute(1,0,2) # [Batch, Output length, Channel]

import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def f_mean(past_activity: np.ndarray) -> np.ndarray:
  """Mean baseline

  Args:
    past_activity: Past activity as time x neurons matrix.

  Returns:
    Predicted activity calculated by taking the per-neuron mean across time and
    repeating it for all 32 timesteps in the prediction horizon.
  """
  return past_activity.mean(axis=0).reshape((1, -1)).repeat(
      constants.PREDICTION_WINDOW_LENGTH, axis=0)

def get_f_model(model: nn.Module):

  def f_model(past_activity: np.ndarray) -> np.ndarray:
    act = torch.from_numpy(past_activity).unsqueeze(1).to(device)
    with torch.no_grad():
      if isinstance(model, POCO):
        pred = model([act])[0]
      else:
        pred = model(act)
    return pred.squeeze(1).cpu().numpy()

  return f_model

"""For inference, we create a data source containing the full trace matrix, and index it as described in [the manuscript](https://openreview.net/pdf?id=oCHsDpyawq) (section 3.2) to compute metrics."""

from collections import defaultdict
from connectomics.jax import metrics
from tqdm import tqdm

def get_infer_source(num_timesteps_context):
  infer_source = data_source.TensorStoreTimeSeries(
      data_source.TensorStoreTimeSeriesConfig(
          input_spec=data_utils.get_spec('240930_traces'),
          timesteps_input=num_timesteps_context,
          timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
      ),
      prefetch=True
  )

  return infer_source

def eval(model: nn.Module, infer_source, num_timesteps_context):
  """Evaluate the model on the test set."""

  model.eval()
  if isinstance(model, nn.Module):
    f = get_f_model(model)
  else:
    f = model

  # Placeholder for results
  MAEs = defaultdict(list)

  # Iterate over all conditions, and make predictions for all contiguous snippets
  # of length 32 in the respective test set.
  for condition_id, condition_name in tqdm(enumerate(constants.CONDITION_NAMES)):
    split = ('test' if condition_id not in constants.CONDITIONS_HOLDOUT
            else 'test_holdout')
    test_min, test_max = data_utils.adjust_condition_bounds_for_split(
        split,
        *data_utils.get_condition_bounds(condition_id),
        num_timesteps_context=num_timesteps_context)

    for window in range(
        data_utils.get_num_windows(test_min, test_max, num_timesteps_context)):
      element = infer_source[test_min + window]

      predictions = f(element['series_input'])
      mae = metrics.mae(predictions=predictions, targets=element['series_output'])

      MAEs[condition_name].append(np.array(mae))

  return MAEs

def get_average(loss_dict: dict):
  return np.mean([np.mean(loss, axis=0) for loss in loss_dict.values()], axis=0)

"""Define the training loop!"""

def train(
  model: nn.Module,
  context_length: int,
  seed: int = 0,
  save_dir: str = None,
  batch_size: int = 4,
  num_epochs: int = 1,
  lr: float = 1e-3,
  loss_fn: str = 'L1Loss',
  weight_decay: float = 0.0,  # NEW
):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
  log_every = 100
  i_iter = 0

  infer_source = get_infer_source(context_length)
  eval_source = get_source(context_length, split='val')
  source = get_source(context_length)
  data_loader = get_data_loader(source, seed=seed, batch_size=batch_size, num_epochs=num_epochs)
  eval_loader = get_data_loader(eval_source, seed=seed, batch_size=batch_size, num_epochs=1)
  val_criterion = torch.nn.L1Loss()

  if loss_fn == 'L1Loss':
    criterion = torch.nn.L1Loss()
  elif loss_fn == 'MSELoss':
    criterion = torch.nn.MSELoss()
  else:
    raise ValueError(f'Loss function {loss_fn} not supported.')

  mv_avg_train_loss = 0
  min_eval_loss = 1e10
  os.makedirs(save_dir, exist_ok=True)

  for element in data_loader:
    model.train()
    x = element['series_input']
    y = element['series_output']
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    x = x.permute(1, 0, 2)
    y = y.permute(1, 0, 2)

    optimizer.zero_grad()
    if isinstance(model, POCO):
      pred = model([x])[0]
    else:
      pred = model(x)
  
    loss = criterion(pred, y)
    loss.backward()
    mv_avg_train_loss = 0.99 * mv_avg_train_loss + 0.01 * loss.item()
    optimizer.step()

    i_iter += 1
    if i_iter % log_every == 0:
      # Evaluate on the validation set
      model.eval()
      with torch.no_grad():
        eval_losses = []
        for element in eval_loader:
          x = element['series_input']
          y = element['series_output']
          x = torch.from_numpy(x).to(device)
          y = torch.from_numpy(y).to(device)
          x = x.permute(1, 0, 2)
          y = y.permute(1, 0, 2)

          if isinstance(model, POCO):
            pred = model([x])[0]
          else:
            pred = model(x)

          eval_losses.append(val_criterion(pred, y).item())
        eval_loss = np.mean(eval_losses)

      print(f'Train iteration {i_iter}, train loss {mv_avg_train_loss:.4f}, eval loss {eval_loss:.4f}')
      if eval_loss < min_eval_loss:
        min_eval_loss = eval_loss
        if save_dir is not None:
          torch.save(model.state_dict(), os.path.join(save_dir, f'best.pt'))

  model.load_state_dict(torch.load(os.path.join(save_dir, f'best.pt')))
  mae = eval(model, infer_source, context_length)
  avg_mae = get_average(mae)

  if save_dir is not None:
    torch.save((mae, avg_mae), os.path.join(save_dir, f'mae.npy'))

  test_result_file = os.path.join(save_dir, f'test_results.txt')
  with open(test_result_file, 'w') as f:
    f.write('Final test set MAE:\n')
    f.write(f'{avg_mae[[0, 1, 3, 7, 15, 31]]}\n')

  print(f'Final test set MAE: {avg_mae[[0, 1, 3, 7, 15, 31]]}')

  return model, mae, avg_mae

"""Check the model performance in Zapbench"""

from connectomics.common import ts_utils
import pandas as pd

# Load dataframe with results reported in the manuscript.
df = pd.DataFrame(
    ts_utils.load_json(f'gs://zapbench-release/dataframes/20250131/combined.json'))
print(df.head())

pred_length = 32
avg_mae = {}
cond_mae = {}

for context in [4, 256]:
  for method in ['mean', 'unet', 'linear']:

    mae_list = []
    for condition_name in constants.CONDITION_NAMES:
      mae_df = df.query(
          f'method == "{method}" and context == {context} and condition == "{condition_name}"'
      ).sort_values('steps_ahead')

      mae = mae_df['MAE'].to_numpy()
      if mae.shape[0] > pred_length:
        mae = mae.reshape(pred_length, -1).mean(axis=1)
      mae_list.append(mae)

    mae_list = np.stack(mae_list)
    avg_mae[f'{method}_{context}'] = mae_list[constants.CONDITIONS_TRAIN, :].mean(axis=0)
    print(method, context, avg_mae[f'{method}_{context}'][[0, 1, 3, 7, 15, 31]], avg_mae[f'{method}_{context}'].mean())

""" get results """

save_dir = "experiments/zapbench"

import os as os
import random
import argparse

# Constants
pred_length = 32
input_size = 71721

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, help="Random seed")
parser.add_argument("--context", type=int, required=True, help="Context length (e.g., 4 or 256)")
parser.add_argument("--model", type=str, required=True, choices=["poco", "linear"], help="Model type to train")
parser.add_argument("--save_dir", type=str, default='experiments/zapbench', help="Directory to save models")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--loss_fn", type=str, default='L1Loss', choices=["L1Loss", "MSELoss"], help="Loss function")
parser.add_argument("--compression_factor", type=int, default=0, help="Compression factor for POCO model")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
parser.add_argument("--poyo_unit_dropout", type=float, default=0.0, help="Dropout for POYO units")
parser.add_argument("--conditioning_dim", type=int, default=1024, help="Conditioning dimension (used in some models)")
parser.add_argument("--decoder_context_length", type=int, default=0, help="Decoder context length (used in some models)")
args = parser.parse_args()

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Prepare config
configs = NeuralPredictionConfig()
configs.seq_length = args.context + pred_length
configs.pred_length = pred_length
configs.conditioning_dim = args.conditioning_dim
configs.poyo_unit_dropout = args.poyo_unit_dropout

if args.compression_factor > 0:
    configs.compression_factor = args.compression_factor
else:
    configs.compression_factor = {4: 4, 48: 16, 256: 64}[args.context]
configs.rotary_attention_tmax = {4: 50, 256: 400}[args.context]

if args.decoder_context_length != 0:
    configs.decoder_context_length = args.decoder_context_length
    assert configs.decoder_context_length % configs.compression_factor == 0

# Prepare model
if args.model == "poco":
    model_instance = POCO(configs, [[input_size]])
elif args.model == "linear":
    model_instance = NLinear(configs, input_size)

# Model directory (include all relevant hyperparameters)
model_dir = os.path.join(
    args.save_dir,
    f"{args.model}_ctx{args.context}_dctx{args.decoder_context_length}_lr{args.lr}_loss{args.loss_fn}"
    f"_cf{configs.compression_factor}_wd{args.weight_decay}_drop{args.poyo_unit_dropout}"
    f"_cond{args.conditioning_dim}_seed{args.seed}"
)
print(f"Model directory: {model_dir}")

# Train model
model, mae, avg_mae = train(
    model_instance,
    args.context,
    seed=args.seed,
    save_dir=model_dir,
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    lr=args.lr,
    loss_fn=args.loss_fn,
    weight_decay=args.weight_decay
)