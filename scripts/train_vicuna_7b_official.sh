#!/bin/bash
#SBATCH -o ./experiments/%j.out
#SBATCH -e ./experiments/%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J eagle
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem 80GB

mkdir ./experiments/$SLURM_JOB_ID
export WANDB_MODE="offline"
accelerate launch -m --mixed_precision=bf16 eagle.train.main \
    --basepath models/vicuna-7b-v1.3 \
    --tmpdir data/eagle-generated-data/sharegpt_0_67999_mufp16 \
    --cpdir test_2_eagle_vicuna-7b-v1.3 \
    --configpath eagle/train/vicuna_7B_config.json