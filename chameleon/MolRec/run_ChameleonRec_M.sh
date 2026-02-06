#!/bin/bash

# sleep 2h

echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain ID --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain WebService --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Medical --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Facility --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Synthetic --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain HumanActivity --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Sensor --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Environment --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Finance --loss_type regression --d_model 64 
CUDA_VISIBLE_DEVICES=0 taskset -c 30-35 python train_M.py --domain Traffic --loss_type regression --d_model 64 

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
CUDA_VISIBLE_DEVICES=0 taskset -c 30 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec --variant ID --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 31 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec --variant OOD --save True