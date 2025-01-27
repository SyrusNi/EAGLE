srun --pty -N 1 -n 1 -p a800 -q normal --gres=gpu:1 --mem 40GB -t 4:00:00 /bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main \
    --basepath models/vicuna-7b-v1.3 \
    --tmpdir data/eagle-generated-data/sharegpt_0_67999_mufp16/0 \
    --cpdir test_eagle_vicuna-7b-v1.3 \
    --configpath eagle/train/vicuna_7B_config.json

python -m eagle.ge_data.allocation --outdir data/eagle-generated-data