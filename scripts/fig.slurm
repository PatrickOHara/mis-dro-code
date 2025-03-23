#!/bin/bash
#
#SBATCH --job-name=fig
#SBATCH --partition=cpu-batch
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=2-00:00:00
#SBATCH --array=0-20
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Loop over each batch ##
python mis-dro-code/scripts/robas_intro_fig.py