#!/bin/bash

echo "Starting training..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/chameleon/MolRec
python train_U.py --domain ID 
python train_U.py --domain WebService 
python train_U.py --domain Medical 
python train_U.py --domain Facility 
python train_U.py --domain Synthetic 
python train_U.py --domain HumanActivity 
python train_U.py --domain Sensor 
python train_U.py --domain Environment 
python train_U.py --domain Finance 
python train_U.py --domain Traffic 

echo "Running AutoAD benchmark..."
cd /data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/benchmark_exp
python run_AutoAD_U.py  --AutoAD_Name ChameleonRec --variant ID --save True &
python run_AutoAD_U.py  --AutoAD_Name ChameleonRec --variant OOD --save True