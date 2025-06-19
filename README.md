### Installation

If you just want to use the POCO model in your own codebase, you can use `<code>zapbench_scripts/standalone_poco.py</code>`, which is a standalone version of the model.

To install dependencies for the codebase (currently only tested on Linux), run:

```bash
conda env create -f environment.yml
conda activate poco
```

Zapbench experiments use a separate set of training and analysis scripts in `<code>zapbench_scripts</code>`.

---

### Datasets

This project uses multiple public neural datasets:

* **Zebrafish** data from [Chen et al., *Neuron*, 2018](https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617). All subjects are used except for subjects 8, 9, and 11.
* **C. elegans** data from Kato et al., *Cell*. Available [here](https://github.com/akshey-kumar/BunDLe-Net/tree/main/data/raw).
* **C. elegans** data from [Atanas & Kim et al., *Cell*, 2023](https://wormwideweb.org/activity/dataset/). We use all 40 NeuroPAL sessions in their dataset.

To set up the neural data, first place the raw data in `data/raw_datasetname/`. See `configs/config_global` for the expected paths for each dataset. Then edit and run `run_preprocess.py` to preprocess the corresponding dataset. Preprocessing includes normalization, PCA, and optional filtering. The processed data will be saved in `data/processed_datasetname`.

You can also generate simulated data by modifying `run_preprocess.py`.

To use your own data, place it in `data/...` and implement a custom data processing pipeline similar to those in `datasets/datasets.py`. In particular, you can inherit from the `NeuralDataset` class and implement the `load_all_activities` method. You’ll also need to define how to initialize the dataset in `datasets/dataloader.py` and add configuration options in `configs/configure_model_datasets`.

---

### Train and Analyze

To run an experiment, use:

```bash
python main.py -t exp_name
```

Each experiment is defined as a function in `configs/experiments.py`, which returns a dictionary of configurations. You can add your own experiment functions to customize hyperparameters, datasets, or model settings. See `configs/configs.py` for a list of predefined hyperparameters. Some parameters are automatically configured based on the dataset and model.

During training, evaluation results and prediction visualizations will be saved in `experiments/exp_name/...`, where you can find training progress, the best saved model, and predictions on the validation set.

If you have multiple GPUs, use the `-s` flag to run multiple experiments concurrently. On a Slurm cluster, use `-c` to submit jobs, and `-p ...` and `--acc ...` to specify the partition and account name.

To analyze results, use:

```bash
python main.py -a exp_name
```

Available analysis functions are listed in `configs/exp_analysis.py`. If the name of the analysis is `x_analysis`, simply run the command above and the corresponding figures will be saved under `figures/...`.

**Example:**
To run an example with 1000 training steps using the zebrafish dataset from Ahrens et al. and the C. elegans dataset from Zimmer et al., run:

```bash
python main.py -t poco_test
python main.py -a poco_test
```

---

### Code Structure

* `configs/configs.py`: Base config, overridden by individual experiments in `configs/experiments.py`.
* `model/model_utils.py`: Defines model types and RNN types.
* `models/multi_session_models.py` and `models/single_session_models.py`: Model implementations.
* `tasks/taskfunction.py`: Implements the neural prediction task.
* `datasets/datasets.py`: Implements dataset loading and train/test splitting.
* `main.py`: Saves configs for each run and submits jobs (if `-c` is used).
* `train.py`: Implements the training loop.

---

### Acknowledgements

This work was supported by the NIH (RF1DA056403), James S. McDonnell Foundation (220020466), Simons Foundation (Pilot Extension-00003332-02), McKnight Endowment Fund, CIFAR Azrieli Global Scholar Program, and NSF (2046583).

Some code snippets are based on the following prior work:

* [POYO](https://poyo-brain.github.io/)
* [TSMixer](https://github.com/ditschuk/pytorch-tsmixer)
* [PLRNN](https://github.com/DurstewitzLab/dendPLRNN)
* [NetFormer](https://github.com/NeuroAIHub/NetFormer)
* [ModernTCN](https://github.com/luodhhh/ModernTCN)
* [FilterNet](https://github.com/aikunyi/FilterNet)