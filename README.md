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

You will also need to install and get a license for the Gurobi optimiser software package. [See this link](https://www.gurobi.com/features/academic-named-user-license/).

### Bayesian DRO

To *only* install the dependencies for the code of Bayesian DRO (Shapiro et al., 2023):

```
pip install -r bayesian_dro/bdro_requirements.txt
```

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