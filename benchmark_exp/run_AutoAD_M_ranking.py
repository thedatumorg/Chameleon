import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, re, logging, pickle
import itertools

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import sys
sys.path.append("/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon")
from chameleon.MolRec.utils import *
from chameleon.ModelOpt.AnoOpt import *

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

# Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
#         'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
#         'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']
Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']

def configure_logger(filename):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def run_AutoAD(model_name, variant, data, Candidate_Model_Set, args, debug_mode, **kwargs):
    if not debug_mode:
        try:
            function_name = f'run_{model_name}'
            function_to_call = globals()[function_name]
            results = function_to_call(variant, data, Candidate_Model_Set, args, **kwargs)
            return results
        except KeyError:
            error_message = f"Model function '{function_name}' is not defined."
            print(error_message)
            return error_message
        except Exception as e:
            error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
            print(error_message)
            return error_message
    else:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(variant, data, Candidate_Model_Set, args, **kwargs)
        return results     

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Automated Solution')
    parser.add_argument('--dataset_dir', type=str, default='/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-M/')
    parser.add_argument('--score_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/TSB-AutoAD/resource/score/multi/')

    parser.add_argument('--file_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-M-Eval.csv')

    parser.add_argument('--pretrained_weights', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/weights/TSB-AD-M/')
    parser.add_argument('--eval_path', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/Candidate/TSB-AD-M/')
    parser.add_argument('--eval_list', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-M-Label.csv')

    parser.add_argument('--AutoAD_Name', type=str, default='SATzilla')
    parser.add_argument('--variant', type=str, default='ID')
    
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/')
    args = parser.parse_args()

    if args.variant is not None:
        Automated_solution = args.AutoAD_Name + '_' + args.variant
    else:
        Automated_solution = args.AutoAD_Name

    # if args.save:
    #     target_dir = os.path.join(args.save_dir, Automated_solution)
    #     os.makedirs(target_dir, exist_ok = True)
    #     logger = configure_logger(filename=f'{target_dir}/000_run_{Automated_solution}.log')

    ## Resume
    if args.resume and os.path.exists(f'{args.save_dir}/{Automated_solution}.csv'):
        df = pd.read_csv(f'{args.save_dir}/{Automated_solution}.csv')
        write_csv =  df.values.tolist()
        Resume_file_name_list = df['file'].tolist()
    else:
        write_csv = []
        Resume_file_name_list = []

    file_list = pd.read_csv(args.file_list)['file_name'].values
    for filename in file_list:
        if filename in Resume_file_name_list: continue
        args.filename = filename
        print('Processing:{} by {}'.format(filename, Automated_solution))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        args.ts_len = len(label)
        print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        ### Start Automated Solution
        start_time = time.time()

        output, flag = run_AutoAD(args.AutoAD_Name, args.variant, data, Candidate_Model_Set, args, debug_mode=not args.save)

        end_time = time.time()
        run_time = end_time - start_time

        if not args.save:
            print('output: ', output)
            print('run_time: ', run_time)

        if args.save:

            list_w = []
            list_w.append(filename)
            list_w.append(run_time)
            list_w.append(output[0])
            list_w.append(output[1])

            write_csv.append(list_w)

            ## Temp Save
            col_w = []
            col_w.append('file')
            col_w.append('Time')
            col_w.append('RankingOrig')
            col_w.append('RankingOpt')

            w_csv = pd.DataFrame(write_csv, columns=col_w)
            w_csv.to_csv(f'{args.save_dir}/{Automated_solution}.csv', index=False)