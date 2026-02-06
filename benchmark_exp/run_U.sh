#!/bin/bash

CUDA_VISIBLE_DEVICES=0 taskset -c 70 python run_AutoAD_U_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant ID --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 71 python run_AutoAD_U_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant OOD --save True

CUDA_VISIBLE_DEVICES=0 taskset -c 76 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant 1 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 77 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant 2 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 78 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant 3 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 79 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant 4 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 80 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_ID --variant 5 --save True &

CUDA_VISIBLE_DEVICES=0 taskset -c 81 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant 1 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 82 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant 2 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 83 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant 3 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 84 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant 4 --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 85 python run_AutoAD_U.py --AutoAD_Name ChameleonOpt_U_OOD --variant 5 --save True &


# CUDA_VISIBLE_DEVICES=0 taskset -c 30 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_ID --variant 1 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 31 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_ID --variant 2 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 32 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_ID --variant 3 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 33 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_ID --variant 4 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 34 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_ID --variant 5 --save True &

# CUDA_VISIBLE_DEVICES=0 taskset -c 35 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_OOD --variant 1 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 36 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_OOD --variant 2 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 37 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_OOD --variant 3 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 38 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_OOD --variant 4 --save True &
# CUDA_VISIBLE_DEVICES=0 taskset -c 39 python run_AutoAD_U.py --AutoAD_Name ChameleonEns_U_OOD --variant 5 --save True &