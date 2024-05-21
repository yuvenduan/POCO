To install dependencies, run:
```bash
conda env create -f environment.yml
conda activate metafish
```

To run an experiment, run:
```bash
python main.py -t exp_name
```
See configs/experiments.py for a list of available experiments. If you have multiple GPUs on your machine, you can add <code>-s</code> so that experiments could be run concurrently. If you are using a slurm cluster, you can add <code>-c</code> so that experiments will be submitted to the cluster and use <code> -p ... </code> and <code> --acc ... </code> to designate partion and account name. During training, loss curves and sample prediction visualization will be saved in <code>experiments/exp_name</code>.

To analyze results, run:
```bash
python main.py -a exp_name
```
See configs/exp_analysis.py for a list of available analyses. Note that if the name of the analysis is <code>x_analysis</code>, you should run <code> python main.py -a exp_name </code>. The figures would be saved in some subfolder of <code>figures</code>.