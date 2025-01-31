#!/bin/bash
#SBATCH -o ./experiments/%j.out
#SBATCH -e ./experiments/%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J eagle
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 8:00:00
#SBATCH --mem 80GB

mkdir ./experiments/$SLURM_JOB_ID
python -m eagle.ge_data.allocation --outdir data/eagle-generated-data