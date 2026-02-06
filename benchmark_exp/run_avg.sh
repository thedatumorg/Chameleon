#!/bin/bash


CUDA_VISIBLE_DEVICES=0 taskset -c 60 python run_AutoAD_U.py --AutoAD_Name Avg_Ens --variant None --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 61 python run_AutoAD_M.py --AutoAD_Name Avg_Ens --variant None --save True &



