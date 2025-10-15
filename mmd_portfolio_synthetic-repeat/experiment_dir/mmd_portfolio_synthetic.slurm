#!/bin/bash
#
#SBATCH --job-name=mmd-newsvendor
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20000
#SBATCH --time=2-00:00:00
#SBATCH --array=0-10
#SBATCH --exclude=sparrow,sparrow-02
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Loop over each batch ##
start=$(($SLURM_ARRAY_TASK_ID * 1))
srun misdro batch /dcs/pg24/u5674159/mis-dro-code/mmd_portfolio_synthetic-repeat/experiment_dir $start 1 --npl-samples-dir /dcs/pg24/u5674159/mis-dro-code/mmd_portfolio_synthetic-repeat/npl_samples_dir