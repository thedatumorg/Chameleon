#!/bin/bash

# sleep 2h

echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain ID --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain WebService --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Medical --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Facility --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Synthetic --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain HumanActivity --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Sensor --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Environment --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Finance --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python train_U.py --domain Traffic --loss_type regression --d_model 512 --precomputed_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/precomputed_resid_processing_flow_stl_ad/

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python run_AutoAD_U.py  --AutoAD_Name ChameleonRec --variant ID --save True &
CUDA_VISIBLE_DEVICES=1 taskset -c 70-75 python run_AutoAD_U.py  --AutoAD_Name ChameleonRec --variant OOD --save True