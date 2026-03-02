#!/bin/bash

echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
python train_M.py --domain ID  
python train_M.py --domain WebService  
python train_M.py --domain Medical  
python train_M.py --domain Facility  
python train_M.py --domain Synthetic  
python train_M.py --domain HumanActivity  
python train_M.py --domain Sensor  
python train_M.py --domain Environment  
python train_M.py --domain Finance  
python train_M.py --domain Traffic  

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
python run_AutoAD_M.py  --AutoAD_Name ChameleonRec --variant ID --save True &
python run_AutoAD_M.py  --AutoAD_Name ChameleonRec --variant OOD --save True