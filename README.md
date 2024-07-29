### Installation

To install dependencies, run (for now the configuration is only tested on Linux):
```bash
conda env create -f environment.yml
conda activate metafish
```

To set up the spontaneous zebrafish data from (Perich et al., 2024), first put raw data from different fish in <code>data/raw</code> (each fish should be a .h5 file). You can use an alternative data path by changing <code>configs/configs_global.py</code>. Then run:
```bash
python run_preprocess.py
```
to normalize calcium traces and do PCA. The processed data will be saved in <code>data/processed</code>. You can also edit <code>run_preprocess.py</code> to generate simulated data.

### Usage

To run an experiment, run:
```bash
python main.py -t exp_name
```
See <code>configs/experiments.py</code> for a list of available experiments. If you have multiple GPUs on your machine, you can add <code>-s</code> so that experiments can be run concurrently. If you are using a Slurm cluster, you can add <code>-c</code> so that experiments will be submitted to the cluster and use <code> -p ... </code> and <code> --acc ... </code> to designate partition and account name. During training, eval results and sample prediction visualization will be saved in <code>experiments/exp_name</code>. 

To analyze results, run:
```bash
python main.py -a exp_name
```
See <code>configs/exp_analysis.py</code> for a list of available analyses. Note that if the name of the analysis is <code>x_analysis</code>, you should run <code> python main.py -a exp_name </code>. The figures would be saved in some subfolder of <code>figures</code>.

As a simple example with 1000 training steps and only one seed, run
```bash
python main.py -t test
python main.py -a test
```

### Code Structure

* <code>configs/configs.py</code> defines the base config, which could be changed in individual experiments in <code>configs/experiments.py</code>.
* <code>model/model_utils.py</code> defines the model types and rnn types used in the experiments. 
* <code>model/model.py</code> implements several model types including autoregressive models and latent dynamics models.
* <code>tasks/taskfunction.py</code> implements the neural prediction task
* <code>datasets/zebrafish.py</code> implements the dataset for neural prediction, including how data is normalized, chunked, and partitioned into training and test sets.
* <code>main.py</code> save config for each run in a experiment (and submit the jobs to cluster if with -c)
* <code>train.py</code> implements the training loop

### Acknowledgements

TODO