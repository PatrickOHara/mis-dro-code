#!/bin/bash
#
#SBATCH --job-name=markowitz
#SBATCH --partition=parrot
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=2-00:00:00
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

# Note that I have removed the array Slurm parameter found in other Slurm scripts in this project

# TODO: Note that there are other import parameters set directly in the script, maybe bring them up here
srun python /dcs/pg24/u5674159/mis-dro-code/markowitz_results_james/main.py