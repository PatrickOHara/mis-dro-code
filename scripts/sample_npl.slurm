#!/bin/bash
#
#SBATCH --job-name=npl # Job name for tracking
#SBATCH --partition=gecko  # Partition you wish to use (see above for list)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=128000            # 60GB RAM
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Execute newsvendor programs ##
srun misdro npl newsvendor bimodal_univariate_gaussian /dcs/large/u1508153/misdro/mmd_newsvendor_1d
# srun misdro npl newsvendor contaminated_exp /dcs/large/u1508153/misdro/mmd_newsvendor_1d
# srun misdro npl newsvendor student_t /dcs/large/u1508153/misdro/mmd_newsvendor_1d

## Execute portfolio programs ##
## srun misdro npl portfolio DowJones /dcs/large/u1508153/misdro/mmd_portfolio --dataset-dir /dcs/pg20/u1508153/datasets/misdro/mmc2
