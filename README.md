### Installation

If you just want to use the POCO model in your own codebase, you can use <code>zapbench_scripts/standalone_poco.py</code>, which is a standalone version for the model.

To install dependencies for the codebase, run (for now the configuration is only tested on Linux):
```bash
conda env create -f environment.yml
conda activate poco
```

Zapbench experiments use a separate set of training and analysis scripts as in <code>zapbench_scripts</code>.

### Datasets

This work uses multiple public neural datasets:

* Zebrafish data from [Chen et al., Neuron 2018](https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617). All subjects except subject 8, 9, 11 are used.
* C. elegans data from Kato et al., Cell. Data can be found at [here](https://github.com/akshey-kumar/BunDLe-Net/tree/main/data/raw). 
* C. elegans data from [Atanas & Kim et al., Cell 2023](https://wormwideweb.org/activity/dataset/). We used all 40 sessions with NeuroPAL in data collected in Atanas & Kim et al..

To set up neural data, first put raw data in <code>data/raw_datasetname/</code>. See <code>configs/config_global</code> for paths for different datasets. Then you can edit and run <code>run_preprocess.py</code> to preprocess the corresponding dataset, preprocessing include normalizing, computing PCs and optional filtering. The processed data will be saved in <code>data/processed_datasetname</code>. You can also edit <code>run_preprocess.py</code> to generate simulated data.

You can also put your own data in <code>data/...</code> and implement your own data processing pipeline like those in <code>datasets/datasets.py</code>. In particular, you can inherit from <code>NeuralDataset</code> class and implement <code>load_all_activities</code>, then define how to initialize the dataset in <code>datasets/dataloader.py</code> and add configs for the dataset in <code>configs/configure_model_datasets</code>.

### Train and Analyze

To run an experiment, run:
```bash
python main.py -t exp_name
```
Each experiment is defined as a function in <code>configs/experiments.py</code> that returns a dictionary of configurations. You can also define your own experiments with customized hyperparameters/datasets/model by adding a function. For a list of predefined hyperparameters, see <code>configs/configs.py</code>. Some parameteres are configured according to the dataset and the model in <code> </code>. During training, eval results and sample prediction visualization will be saved in <code>experiments/exp_name/...</code>, where you can find training progress, saved best model, and sample predictions on the validation set.

If you have multiple GPUs on your machine, you can add <code>-s</code> so that different runs can be run concurrently. If you are using a Slurm cluster, you can add <code>-c</code> so that jobs will be submitted to the cluster and use <code> -p ... </code> and <code> --acc ... </code> to designate partition and account name. 

To analyze results, run:
```bash
python main.py -a exp_name
```
See <code>configs/exp_analysis.py</code> for a list of available analyses. Note that if the name of the analysis is <code>x_analysis</code>, you should run <code>python main.py -a exp_name</code>. The figures would be saved in some subfolder of <code>figures</code>.

As an example with 1000 training steps using the zebrafish dataset from Ahrens et al. and the C. elegans dataset from Zimmer et al, run
```bash
python main.py -t test
python main.py -a test
```

### Code Structure

* <code>configs/configs.py</code> defines the base config, which could be changed in individual experiments in <code>configs/experiments.py</code>.
* <code>model/model_utils.py</code> defines the model types and rnn types used in the experiments. 
* <code>models/multi_session_models.py</code> and <code> models/single_session_models.py </code> implement the models.
* <code>tasks/taskfunction.py</code> implements the neural prediction task.
* <code>datasets/datasets.py</code> implements the dataset for neural prediction, partition into training and test sets.
* <code>main.py</code> save config for each run in a experiment (and submit the jobs to cluster if with -c)
* <code>train.py</code> implements the training loop

### Acknowledgements

This work was supported by the NIH (RF1DA056403), James S. McDonnell Foundation (220020466), Simons Foundation (Pilot Extension-00003332-02), McKnight Endowment Fund, CIFAR Azrieli Global Scholar Program, and NSF (2046583).

Some code snipets are based on previous work:
* POYO: https://poyo-brain.github.io/
* TSMixer: https://github.com/ditschuk/pytorch-tsmixer
* PLRNN: https://github.com/DurstewitzLab/dendPLRNN
* NetFormer: https://github.com/NeuroAIHub/NetFormer
* ModernTCN: https://github.com/luodhhh/ModernTCN
* FilterNet: https://github.com/aikunyi/FilterNet
