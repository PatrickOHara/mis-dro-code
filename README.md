# mis-dro-code
Misspecified DRO code

## Installation

First use conda (or pyenv) to create a new env, e.g.:

```
conda create --name mis-dro -c conda-forge python=3.11
```

Next, install the mis_dro and bayesian_dro packages with pip.
This can be done with one pip command:
```
cd mis-dro-code
pip install -e .
```

You will also need to install and get a license for:
- the [Gurobi optimiser software package](https://www.gurobi.com/features/academic-named-user-license/).
- the [MOSEK conic optimisation solver](https://www.mosek.com/products/academic-licenses/).

### Bayesian DRO

To *only* install the dependencies for the code of Bayesian DRO (Shapiro et al., 2023):

```
pip install -r bayesian_dro/bdro_requirements.txt
```

***

## How to run

To run just one algorithm, you can use the CLI `misdro` with the `run` command:
```
misdro run --num-posterior-samples 100 --posterior bayes --num-observations 10
```

To setup an experiment, including a custom SLURM file, you can use the `misdro setup` commmand. For example:
```
misdro setup $EXPERIMENT_DIR
```
where `$EXPERIMENT_DIR` is the filepath to a directory you want to store the experiment inside.
A SLURM file is generated for each dataset.
Each SLURM file will be put inside `$EXPERIMENT_DIR` along with a `experiment.json` file containing all of the parameters for the experiment.
You can run a slurm file:

```
sbatch NAME.slurm
```

and the results for each parameter configuration will be saved in a CSV file with a unique UUID.
To collect all of the CSV files together in a single `results.csv` file, you can run:
```
misdro csv $EXPERIMENT_DIR
```
then you can analyse the results using the `newsvendor_experiment.ipynb` notebook.

### Sampling from the NPL

Setup the MMD experiment. You will need to pass the `experiment_dir` and `npl_samples_dir`.
```
misdro setup-mmd mmd_newsvendor_1d {experiment_dir} {npl_samples_dir} 1
```
If `npl_samples_dir` does not exist, then `setup-mmd` will create a SLURM file and `npl_settings.csv` inside the newly created `npl_samples_dir` ready for you to run the NPL. If `npl_samples_dir` exists, then we assume you are using NPL samples from a previous run. Either way, the `npl_samples_dir` will be passed to the SLURM files so they know where the samples are!

To run NPL samples on the GPU, run the SLURM file using `sbatch`:
```
sbatch sample_npl_mmd_newsvendor_1d.slurm 
```
The above SLURM script will generate a directory for each NPL 'setting'. Inside each directory will be a CSV file for each replication.

Once all the NPL samples have been generated, run the SLURM file from `experiment_dir` using `sbatch`.
```
sbatch mmd_newsvendor_1d.slurm
```


### Bayesian DRO

To run Bayesian DRO experiments with *continuous* support from Section 4 of Shapiro et al (2023):

```
python bayesian_dro/Bayesian_DRO_continuous.py
```

To run Bayesian DRO experiments with *finite* support from Appendix A of Shapiro et al (2023):

```
python bayesian_dro/Bayesian_DRO_finite.py
```

## Testing

You can run the tests with `pytest`. To install the test dependencies:
```
pip install -r requirements.txt
```
To run the tests:
```
pytest tests
```

### Formatting and Linting

You might also like to make your code look pretty with the `black` formatter:
```
black */
```

## References

Shapiro, A., Zhou, E., & Lin, Y. (2023). Bayesian distributionally robust optimization. SIAM Journal on Optimization, 33(2), 1279-1304.