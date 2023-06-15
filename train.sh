#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_Train_$2_$1_$now"
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --archs $1 --benchmark $2 $4 2>&1|tee logs/$log_name.log

# bash train.sh TSA FineDiving 0,1 [--resume]
