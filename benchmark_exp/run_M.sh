#!/bin/bash

CUDA_VISIBLE_DEVICES=1 taskset -c 72 python run_AutoAD_M_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant ID --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 73 python run_AutoAD_M_ranking.py --AutoAD_Name ChameleonOpt_precomputed --variant OOD --save True

CUDA_VISIBLE_DEVICES=1 taskset -c 86 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant 1 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 87 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant 2 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 88 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant 3 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 89 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant 4 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 90 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_ID --variant 5 --save True &

CUDA_VISIBLE_DEVICES=1 taskset -c 91 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant 1 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 92 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant 2 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 93 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant 3 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 94 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant 4 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 95 python run_AutoAD_M.py --AutoAD_Name ChameleonOpt_M_OOD --variant 5 --save True &


CUDA_VISIBLE_DEVICES=1 taskset -c 40 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_ID --variant 1 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 41 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_ID --variant 2 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 42 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_ID --variant 3 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 43 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_ID --variant 4 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 44 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_ID --variant 5 --save True &

CUDA_VISIBLE_DEVICES=1 taskset -c 45 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_OOD --variant 1 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 46 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_OOD --variant 2 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 47 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_OOD --variant 3 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 48 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_OOD --variant 4 --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 49 python run_AutoAD_M.py --AutoAD_Name ChameleonEns_M_OOD --variant 5 --save True &