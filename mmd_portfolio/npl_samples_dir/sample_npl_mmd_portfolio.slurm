#!/bin/bash
#
#SBATCH --job-name=npl # Job name for tracking
#SBATCH --partition=wmlg-ada  # Partition you wish to use (see above for list)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --array=0-0
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Execute newsvendor programs ##
srun misdro npl /dcs/pg24/u5674159/mis-dro-code/mmd_portfolio/npl_samples_dir $SLURM_ARRAY_TASK_ID --dataset-dir ~/datasets/misdro/mmc2