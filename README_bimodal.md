# Harita's instructions

### To run Gaussian bimodal examples from bimodal branch:

1) First you need to setup two separate experiments:
- ```mmd_newsvendor_exp_1d``` runs both univariate and multivariate (5d) Gaussian for RoBAS and empirical MMD
-  ```kl_newsvendor_exp_1d``` runs both univariate and multivariate (5d) Gaussian for BAS, BDRO and BDRO with NPL posterior
Any of the parameters that needs to be changed can be changed inside these two experiments

2) You further need the NPL posterior samples (which I will send you) in the right folder as specified in ```main.py``` lines 353 and 355

3) To get results and plot use the notebook: bimodal_processing_results.ipynb in the notebooks folder

### If you need to retrieve my results for whaterver reason these are in:
- ```./misdro/results_kdro/slurm_batches/bimodal_robas``` (robas and empirical)
- ```./misdro/results_kdro/slurm_batches/bimodal_bas_bdro``` (bas, bdro, bdro-npl)
