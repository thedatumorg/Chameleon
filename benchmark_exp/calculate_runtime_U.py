import pandas as pd
import math
import re
import os, pickle
from pathlib import Path
import pycatch22 as catch22
import numpy as np
from bidict import bidict
import ast

eva_filesList = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-U-Eval.csv'
Automated_solution_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-U/'
Candidate_path = '/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/TSB-AutoAD/resource/run_time/uni/'
data_direc = '/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-U/'
save_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/runtime/Chameleon_InferenceTime_U.csv'
# save_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/runtime/Chameleon_SelectionTime_U.csv'

ranking_path_ID = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-U/ChameleonOpt_precomputed_ID.csv'
ranking_path_OOD = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-U/ChameleonOpt_precomputed_OOD.csv'
selection_time_path_ID = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-U/ChameleonRec_ID.csv'
selection_time_path_OOD = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-U/ChameleonRec_OOD.csv'

Candidate_Model_Set = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'POLY', 'MatrixProfile', 'KShapeAD', 'SAND', 'Series2Graph', 'SR', 'Sub_PCA', 'Sub_HBOS', 'Sub_OCSVM', 
        'Sub_MCD', 'Sub_KNN', 'KMeansAD_U', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 
        'TimesNet', 'FITS', 'OFA', 'Lag_Llama', 'Chronos', 'TimesFM', 'MOMENT_ZS', 'MOMENT_FT']


Method_list = ['ChameleonRec_ID', 'ChameleonRec_OOD',   # SelectionTime + 1 DetTime
                'Avg_Ens',
                # SelectionTime + Multiple DetTime
               'ChameleonEns_U_ID_1', 'ChameleonEns_U_ID_2', 'ChameleonEns_U_ID_3', 'ChameleonEns_U_ID_4', 'ChameleonEns_U_ID_5', 
               'ChameleonEns_U_OOD_1', 'ChameleonEns_U_OOD_2', 'ChameleonEns_U_OOD_3', 'ChameleonEns_U_OOD_4', 'ChameleonEns_U_OOD_5', 
                # RankingTime + Multiple DetTime
               'ChameleonOpt_U_ID_1', 'ChameleonOpt_U_ID_2', 'ChameleonOpt_U_ID_3', 'ChameleonOpt_U_ID_4', 'ChameleonOpt_U_ID_5', 
               'ChameleonOpt_U_OOD_1', 'ChameleonOpt_U_OOD_2', 'ChameleonOpt_U_OOD_3', 'ChameleonOpt_U_OOD_4', 'ChameleonOpt_U_OOD_5', 
            ]

Method_list_name_mapping = [
    'Chameleon-Rec (ID)', 'Chameleon-Rec (OOD)',
    'AvgEns',
    'Chameleon-Ens-1 (ID)', 'Chameleon-Ens-2 (ID)', 'Chameleon-Ens-3 (ID)', 'Chameleon-Ens-4 (ID)', 'Chameleon-Ens-5 (ID)', 
    'Chameleon-Ens-1 (OOD)', 'Chameleon-Ens-2 (OOD)', 'Chameleon-Ens-3 (OOD)', 'Chameleon-Ens-4 (OOD)', 'Chameleon-Ens-5 (OOD)', 
    'Chameleon-1 (ID)', 'Chameleon-2 (ID)', 'Chameleon-3 (ID)', 'Chameleon-4 (ID)', 'Chameleon-5 (ID)', 
    'Chameleon-1 (OOD)', 'Chameleon-2 (OOD)', 'Chameleon-3 (OOD)', 'Chameleon-4 (OOD)', 'Chameleon-5 (OOD)', 
]

Method_list_name_dict = bidict(dict(zip(Method_list, Method_list_name_mapping)))
print('Total number of automated solution: ', len(Method_list))

def categorize_method(name: str) -> str:
    if 'ChameleonRec' in name:
        return 'Rec'                 # SelectionTime + 1 DetTime
    elif 'ChameleonOpt' in name:
        return 'Opt_logit'           # RankingTime + Multiple DetTime
    elif 'ChameleonEns' in name:
        return 'Opt'                 # SelectionTime + Multiple DetTime
    else:
        return 'Other'               # AvgEns

from collections import defaultdict

groups = defaultdict(list)
for m in Method_list:
    groups[categorize_method(m)].append(m)  # groups['Rec'], groups['Opt'], groups['Opt_logit']


data_write_list = []
file_list = pd.read_csv(eva_filesList)['file_name'].values

for filename in file_list:
    print('filename: ', filename)
    # File Name
    
    for method in Method_list:
        data_write = []
        data_write.append(filename)

        file_path = os.path.join(data_direc, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        data_write.append(len(label))
        data_write.append(Method_list_name_dict[method])


        df_SelectionTime_ID = pd.read_csv(selection_time_path_ID)
        SelectionTime_ID = df_SelectionTime_ID.loc[df_SelectionTime_ID['file'] == filename, 'Time'].values[0]
        df_SelectionTime_OOD = pd.read_csv(selection_time_path_OOD)
        SelectionTime_OOD = df_SelectionTime_OOD.loc[df_SelectionTime_OOD['file'] == filename, 'Time'].values[0]
        
        df_RankingTime_ID = pd.read_csv(ranking_path_ID)
        RankingTime_ID = df_RankingTime_ID.loc[df_RankingTime_ID['file'] == filename, 'Time'].values[0]
        df_RankingTime_OOD = pd.read_csv(ranking_path_OOD)
        RankingTime_OOD = df_RankingTime_OOD.loc[df_RankingTime_OOD['file'] == filename, 'Time'].values[0]

        # AutoML T
        if method in groups['Rec']:
            if 'ID' in method:
                T = SelectionTime_ID
                s = df_RankingTime_ID[df_RankingTime_ID['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:1]
            else:
                T = SelectionTime_OOD
                s = df_RankingTime_OOD[df_RankingTime_OOD['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:1]
            method_name = method

        elif method in groups['Opt']:
            num_models = method.split('_')[-1]
            if 'ID' in method:
                T = SelectionTime_ID
                s = df_RankingTime_ID[df_RankingTime_ID['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]
            else:
                T = SelectionTime_OOD
                s = df_RankingTime_OOD[df_RankingTime_OOD['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]            
            
        elif method in groups['Opt_logit']:
            num_models = method.split('_')[-1]
            if 'ID' in method:
                T = RankingTime_ID
                s = df_RankingTime_ID[df_RankingTime_ID['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]
            else:
                T = RankingTime_OOD
                s = df_RankingTime_OOD[df_RankingTime_OOD['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]  
        else:
            T = 0
            selected_models = Candidate_Model_Set

        # Candidate T
        Candidate_T = 0
        for Candidate_Model in selected_models:
            try:
                path = Candidate_path+Candidate_Model+'.csv'
                df_det = pd.read_csv(path)
                Candidate_T += df_det.loc[df_det['File Name'] == filename, 'Time Cost'].values[0]
            except:
                print('No Candidate Model Time File: ', Candidate_Model)
                Candidate_T = 0

        df_acc = pd.read_csv(Automated_solution_path+method+'.csv')
        data_write.append(df_acc.loc[df_acc['file'] == filename, 'VUS-PR'].values[0])

        Metric = T + Candidate_T
        # Metric = T
        data_write.append(Metric)
            
        data_write_list.append(data_write)

columns = []
columns.append('file')
columns.append('length')
columns.append('method')
columns.append('VUS-PR')
columns.append('runtime')
eval_csv = pd.DataFrame(data_write_list, columns=columns)
eval_csv.to_csv(save_path, index=False)

print(eval_csv)