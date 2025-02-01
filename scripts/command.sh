srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 --mem 40GB -t 4:00:00 /bin/bash

accelerate launch -m --num_processes=4 --mixed_precision=bf16 eagle.train.main \
    --basepath models/vicuna-7b-v1.3 \
    --tmpdir data/eagle-generated-data/sharegpt_0_67999_mufp16 \
    --cpdir test_eagle_vicuna-7b-v1.3 \
    --configpath eagle/train/vicuna_7B_config.json \
    --bs 8 \
    --gradient-accumulation-steps 4 \
    --ckptdir test_eagle_vicuna-7b-v1.3/state_1

python -m eagle.ge_data.allocation --outdir data/eagle-generated-data

# evaluation
python -m eagle.evaluation.gen_ea_answer_vicuna \
    --model-id eagle_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3

python -m eagle.evaluation.gen_ea_answer_vicuna \
    --model-id eagle_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3 \
    --temperature 0

python -m eagle.evaluation.gen_baseline_answer_vicuna \
    --model-id baseline_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3\

python -m eagle.evaluation.gen_baseline_answer_vicuna \
    --model-id baseline_vicuna-7b-v1.3 \
    --ea-model-path models/EAGLE-Vicuna-7B-v1.3 \
    --base-model-path models/vicuna-7b-v1.3 \
    --temperature 0

python -m eagle.evaluation.gen_ea_answer_vicuna \
    --model-id eagle_train_vicuna-7b-v1.3 \
    --ea-model-path test_eagle_vicuna-7b-v1.3 \
    --base-model-path models/vicuna-7b-v1.3