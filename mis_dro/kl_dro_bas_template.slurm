#!/bin/bash
#
#SBATCH --job-name=kl-newsvendor
#SBATCH --partition=parrot
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=2-00:00:00
#SBATCH --array=0-{num_batches}
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Loop over each batch ##
start=$(($SLURM_ARRAY_TASK_ID * 1))
srun misdro batch {experiment_dir} $start {batch_size}
