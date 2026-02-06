import pandas as pd
import math
import re
import os, pickle
from pathlib import Path
import pycatch22 as catch22
import numpy as np
from bidict import bidict
import ast

eva_filesList = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/file_list/TSB-AD-M-Eval.csv'
Automated_solution_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/'
Candidate_path = '/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/TSB-AutoAD/resource/run_time/multi/'
data_direc = '/data/liuqinghua/code/ts/public_repo/TSB-AD/Datasets/TSB-AD-Datasets/TSB-AD-M/'
# save_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/runtime/Chameleon_InferenceTime_M.csv'
save_path = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/runtime/Chameleon_SelectionTime_M.csv'

ranking_path_ID = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/Synthetic_Opt_TopK_precomputed_M_logit_ID.csv'
ranking_path_OOD = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/Synthetic_Opt_TopK_precomputed_M_logit_OOD.csv'
selection_time_path_ID = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/Chameleon_Rec_ID.csv'
selection_time_path_OOD = '/data/liuqinghua/code/ts/TSAD-AutoML/Chameleon/testbed/eval/AutoAD/TSB-AD-M/Chameleon_Rec_OOD.csv'

Candidate_Model_Set = ['IForest', 'LOF', 'PCA', 'HBOS', 'OCSVM', 'MCD', 'KNN', 'KMeansAD', 'COPOD', 'CBLOF', 'EIF', 'RobustPCA', 'AutoEncoder', 
                    'CNN', 'LSTMAD', 'TranAD', 'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'Donut', 'TimesNet', 'FITS', 'OFA']

Method_list = ['Chameleon_Rec_ID', 'Chameleon_Rec_OOD',   # SelectionTime + 1 DetTime
                # SelectionTime + Multiple DetTime
               'Chameleon_Opt_M_1_ID', 'Chameleon_Opt_M_2_ID', 'Chameleon_Opt_M_3_ID', 'Chameleon_Opt_M_4_ID', 'Chameleon_Opt_M_5_ID', 
            #    'Chameleon_Opt_M_7_ID', 'Chameleon_Opt_M_9_ID', 'Chameleon_Opt_M_11_ID',
               'Chameleon_Opt_M_1_OOD', 'Chameleon_Opt_M_2_OOD', 'Chameleon_Opt_M_3_OOD', 'Chameleon_Opt_M_4_OOD', 'Chameleon_Opt_M_5_OOD', 
            #    'Chameleon_Opt_M_7_OOD', 'Chameleon_Opt_M_9_OOD', 'Chameleon_Opt_M_11_OOD',
                # RankingTime + Multiple DetTime
               'Chameleon_Opt_logit_M_1_ID', 'Chameleon_Opt_logit_M_2_ID', 'Chameleon_Opt_logit_M_3_ID', 'Chameleon_Opt_logit_M_4_ID', 'Chameleon_Opt_logit_M_5_ID', 
            #    'Chameleon_Opt_logit_M_7_ID', 'Chameleon_Opt_logit_M_9_ID', 'Chameleon_Opt_logit_M_11_ID',
               'Chameleon_Opt_logit_M_1_OOD', 'Chameleon_Opt_logit_M_2_ODD', 'Chameleon_Opt_logit_M_3_ODD', 'Chameleon_Opt_logit_M_4_ODD', 'Chameleon_Opt_logit_M_5_ODD', 
            #    'Chameleon_Opt_logit_M_7_ODD', 'Chameleon_Opt_logit_M_9_ODD', 'Chameleon_Opt_logit_M_11_ODD'
            ]
Method_list_name_mapping = [
    'Chameleon-Rec (ID)', 'Chameleon-Rec (OOD)',
    'Chameleon-Ens-1 (ID)', 'Chameleon-Ens-2 (ID)', 'Chameleon-Ens-3 (ID)', 'Chameleon-Ens-4 (ID)', 'Chameleon-Ens-5 (ID)', 
    # 'Chameleon-Ens-7 (ID)', 'Chameleon-Ens-9 (ID)', 'Chameleon-Ens-11 (ID)',
    'Chameleon-Ens-1 (OOD)', 'Chameleon-Ens-2 (OOD)', 'Chameleon-Ens-3 (OOD)', 'Chameleon-Ens-4 (OOD)', 'Chameleon-Ens-5 (OOD)', 
    # 'Chameleon-Ens-7 (OOD)', 'Chameleon-Ens-9 (OOD)', 'Chameleon-Ens-11 (OOD)',
    'Chameleon-1 (ID)', 'Chameleon-2 (ID)', 'Chameleon-3 (ID)', 'Chameleon-4 (ID)', 'Chameleon-5 (ID)', 
    # 'Chameleon-7 (ID)', 'Chameleon-9 (ID)', 'Chameleon-11 (ID)',
    'Chameleon-1 (OOD)', 'Chameleon-2 (OOD)', 'Chameleon-3 (OOD)', 'Chameleon-4 (OOD)', 'Chameleon-5 (OOD)', 
    # 'Chameleon-7 (OOD)', 'Chameleon-9 (OOD)', 'Chameleon-11 (OOD)'
]
Method_list_name_dict = bidict(dict(zip(Method_list, Method_list_name_mapping)))
print('Total number of automated solution: ', len(Method_list))

def categorize_method(name: str) -> str:
    if '_Rec_' in name:
        return 'Rec'                 # SelectionTime + 1 DetTime
    elif '_Opt_logit_' in name:
        return 'Opt_logit'           # RankingTime + Multiple DetTime
    elif '_Opt_' in name:
        return 'Opt'                 # SelectionTime + Multiple DetTime
    else:
        raise ValueError(f'Unknown method type: {name}')

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
            num_models = method.split('_')[-2]
            if 'ID' in method:
                T = SelectionTime_ID
                method_name = 'Chameleon_Opt_ID/'+method[:-3]
                s = df_RankingTime_ID[df_RankingTime_ID['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]
            else:
                T = SelectionTime_OOD
                method_name = 'Chameleon_Opt_OOD/'+method[:-4]
                s = df_RankingTime_OOD[df_RankingTime_OOD['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]            
            
        elif method in groups['Opt_logit']:
            num_models = method.split('_')[-2]
            if 'ID' in method:
                T = RankingTime_ID
                method_name = 'Chameleon_Opt_ID/'+method[:-3]
                s = df_RankingTime_ID[df_RankingTime_ID['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]
            else:
                T = RankingTime_OOD
                method_name = 'Chameleon_Opt_OOD/'+method[:-4]
                s = df_RankingTime_OOD[df_RankingTime_OOD['file'] == filename]['RankingOpt'].to_list()[0]
                selected_models = ast.literal_eval(s)[:int(num_models)]  

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

        df_acc = pd.read_csv(Automated_solution_path+method_name+'.csv')
        data_write.append(df_acc.loc[df_acc['file'] == filename, 'VUS-PR'].values[0])

        # Metric = T + Candidate_T
        Metric = T
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