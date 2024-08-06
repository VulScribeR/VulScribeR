#!/bin/bash

# help
# $1 -> model number
# $2 testing set in jsonl format
# sh run-test.sh 1 testing_set.jsonl 0 e.g. sh run-test.sh 10 ../../bigvul-cleaned/final_only/test.jsonl 0
# $3 -> gpu
# SELECT BEST MODEL FIRST!

LR=2e-5
TRAIN_BS=16
VAL_BS=16

# have to set some dummy training set to load the model but set epochs to 0 so it wouldn't be used
CUDA_VISIBLE_DEVICES=$3, python models/linevul_run.py \
    --output_dir=./linevul/origin/saved_models_llm_"$1" \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --evaluate_during_training \
    --eval_data_file=../../bigvul-cleaned/final_only/val.jsonl \
    --train_data_file=../../bigvul-cleaned/final_only/val.jsonl \
    --test_data_file=$2 \
    --epoch 0 \
    --block_size 512 \
    --train_batch_size $TRAIN_BS \
    --eval_batch_size 16 \
    --learning_rate $LR \
    --max_grad_norm 1.0 \
    --resume_training
    