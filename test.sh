#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_Test_$2_$1_$now"
CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --archs $1 --benchmark $2 --test --ckpts $4 2>&1|tee logs/$log_name.log

# bash test.sh TSA FineDiving 0,1 ./experiments/TSA/FineDiving/default/last.pth
