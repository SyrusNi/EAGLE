#!/bin/bash
#SBATCH -o ./experiments/%j.out
#SBATCH -e ./experiments/%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J eagle
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --mem 40GB

mkdir ./experiments/$SLURM_JOB_ID

python -m eagle.evaluation.gen_ea_answer_vicuna \
    --model-id eagle-2_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3 \
    --temperature 0

python -m eagle.evaluation.gen_ea_answer_vicuna \
    --model-id eagle-2_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3 \
    --temperature 1.0