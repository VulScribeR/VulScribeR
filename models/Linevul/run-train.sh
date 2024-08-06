#!/bin/bash

# help
# $1 -> model number
# $2 -> representation level data folder name
# $3 -> GPU(s)
# nohup sh run-train.sh 1 train_set.jsonl 0 &> llm_logs_1.out &


LR=2e-5
TRAIN_BS=16

echo "$1 $2 $3  $TRAIN_BS $LR"

CUDA_VISIBLE_DEVICES=$3, python models/linevul_run.py \
    --output_dir=./linevul/origin/saved_models_llm_"$1" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --evaluate_during_training \
    --eval_data_file=../../bigvul-cleaned/final_only/val.jsonl \
    --train_data_file=../../llm-vulgen/container_data/$2 \
    --test_data_file=../../bigvul-cleaned/final_only/test.jsonl \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size $TRAIN_BS \
    --eval_batch_size 16 \
    --learning_rate $LR \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 123456


echo "DONE! -> find the best model by finding the best F1-score achieving checkpoint, rename it to model.out and redo the test"
