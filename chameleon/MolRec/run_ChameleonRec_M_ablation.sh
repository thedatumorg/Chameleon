#!/bin/bash


echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
CUDA_VISIBLE_DEVICES=0 taskset -c 35-38 python train_M_ablation.py --domain ID --loss_type regression --d_model 64 --backbone_typ ConvNet &
CUDA_VISIBLE_DEVICES=0 taskset -c 38-40 python train_M_ablation.py --domain Medical --loss_type regression --d_model 64 --backbone_typ ConvNet  

CUDA_VISIBLE_DEVICES=0 taskset -c 35-38 python train_M_ablation.py --domain Facility --loss_type regression --d_model 64 --backbone_typ ConvNet  &
CUDA_VISIBLE_DEVICES=0 taskset -c 38-40 python train_M_ablation.py --domain HumanActivity --loss_type regression --d_model 64 --backbone_typ ConvNet  

CUDA_VISIBLE_DEVICES=0 taskset -c 35-38 python train_M_ablation.py --domain Sensor --loss_type regression --d_model 64 --backbone_typ ConvNet  &
CUDA_VISIBLE_DEVICES=0 taskset -c 38-40 python train_M_ablation.py --domain Environment --loss_type regression --d_model 64  --backbone_typ ConvNet

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
CUDA_VISIBLE_DEVICES=0 taskset -c 32 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec_Ablation --variant ID --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 33 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec_Ablation --variant OOD --save True



# None Decomp
echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
CUDA_VISIBLE_DEVICES=0 taskset -c 40-44 python train_M_ablation_decomp.py --domain ID --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/ &
CUDA_VISIBLE_DEVICES=0 taskset -c 44-45 python train_M_ablation_decomp.py --domain Medical --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/

CUDA_VISIBLE_DEVICES=0 taskset -c 40-44 python train_M_ablation_decomp.py --domain Facility --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/  &
CUDA_VISIBLE_DEVICES=0 taskset -c 44-45 python train_M_ablation_decomp.py --domain HumanActivity --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/ 
CUDA_VISIBLE_DEVICES=0 taskset -c 40-44 python train_M_ablation_decomp.py --domain Sensor --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/  &
CUDA_VISIBLE_DEVICES=0 taskset -c 44-45 python train_M_ablation_decomp.py --domain Environment --loss_type regression --d_model 64 --save_dir /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/ChameleonRec_Ablation_decomp/

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
CUDA_VISIBLE_DEVICES=0 taskset -c 40 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec_Ablation_decomp --variant ID --save True &
CUDA_VISIBLE_DEVICES=0 taskset -c 41 python run_AutoAD_M.py  --AutoAD_Name ChameleonRec_Ablation_decomp --variant OOD --save True