# Improving Screening Processes via Calibrated Subset Selection

This repo contains the code for the empirical evaluation in the paper 
[Improving Screening Processes via Calibrated Subset Selection](https://arxiv.org/abs/2202.01147), 
which includes an implementation of the Calibrated Subset Selection algorithm proposed in the paper. 


### Create Environment
Make sure [conda](https://docs.conda.io/en/latest/) is installed. Run
```angular2html
conda env create -f environment.yml
source activate alg_screen
```

### Download and Prepare Data

Set prepare_data = True and submit = False in params_exp_noise.py and params_exp_diversity_noise.py

Run
```angular2html
python ./scripts/run_exp_noise.py
python ./scripts/run_exp_diversity_noise.py
```

### Run Experiments
Set prepare_data = False and submit = True in params_exp_noise.py and params_exp_diversity_noise.py

On a cluster with [Slurm](https://slurm.schedmd.com/documentation.html) workload manager, run
```angular2html
python ./scripts/run_exp_noise.py
python ./scripts/run_exp_cal_size.py
python ./scripts/run_exp_diversity_noise.py
```

### Plot Figures
Run
```angular2html
python ./scripts/plot_exp_normal.py
python ./scripts/plot_exp_diversity.py
```

### Bibtex
```angular2html
@InProceedings{wang/etal/2022/improving,
  title = {Improving Screening Processes via Calibrated Subset Selection},
  author = {Wang, Lequn and Joachims, Thorsten and Gomez-Rodriguez, Manuel},
  booktitle = {International Conference on Machine Learning (ICML)},
  year= {2022}
}
```
