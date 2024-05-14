#!/bin/bash
lang=$1
model_path=$2
peft=$3
tasks=arc_${lang},hellaswag_${lang},mmlu_${lang},truthfulqa_${lang}
tasks=mmlu_${lang}
device=cuda

echo ${peft}
python -u main.py \
    --tasks=${tasks} \
    --model_args pretrained=${model_path},peft=${peft} \
    --device=${device} \
    --batch_size=8
