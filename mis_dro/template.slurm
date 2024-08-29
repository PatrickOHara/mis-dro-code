#!/bin/bash
#
#SBATCH --job-name=newsvendor  # Job name for tracking
#SBATCH --partition=parrot     # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=24     # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem=120000            # 60GB RAM
#SBATCH --exclusive=mcs        # Exclusive mode, only this job will run
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Execute your program(s) ##
srun misdro experiment {experiment_dir} {dgp} {algorithm} --njobs {njobs}
