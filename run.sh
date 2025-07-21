#!/bin/bash

# Bash script to launch the Phoenix-2014 experiment

python main.py \
    --seq 01April_2010_Thursday_heute_default-8 \
    --split train \
    --models slowfast twostream \
    --slowfast-ckpt ../checkpoints/slowfast_phoenix2014_dev_18.01_test_18.28.pt \
    --twostream-ckpt ../checkpoints/twostreamslr.ckpt \
    --target-from-gt \
    --device cuda:0
