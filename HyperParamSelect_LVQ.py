import numpy as np
import pandas as pd
import sys
import os.path
import time, itertools, re
import importlib
from collections import Counter
import scipy as sc
import miceforest as mf
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
print(os.getcwd())
fileloc_data='/'.join(os.getcwd().split('/')[0:5])+ '/data/annonymizedDatasets/'
savetag='pred_lavSQ_MHC'
code_path='/'.join(os.getcwd().split('/')[0:4])+'/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ
from EDdataset_GGZ import colsTypeCast    
df_all_combo=pd.read_csv(fileloc_data+'maskedDAIsy_MainDect_ED_SQ48_MHC_Honos_Lav.tsv', sep='\t', decimal=',')
df_adapted_combo, colsExtracted, subscales=colsTypeCast(df_all_combo)
from HypOpt import GridSearch_LVQ

def LVQ_Hyperparameters(X,Y):
    sampling_strategy='not majority'
    if len(np.unique(Y))==5:
        param_grid_gmlvq={'prot_choice': [np.array([3,3,3,1,1]),np.array([3,2,2,1,1]),
            np.array([2,2,2,1,1]),np.array([1,1,1,1,1]), np.array([1,1,2,1,1])],
            'n_comp':[5,10, 20,26], 'scorer_name':'Lowest_CWA', 'dist': "adaptive-squared-euclidean"}
        param_grid_lgmlvq={'prot_choice': [np.array([3,3,3,1,1]),np.array([3,2,2,1,1]),
            np.array([2,2,2,1,1]),np.array([1,1,1,1,1]), np.array([1,1,2,1,1])],
            'n_comp':[5,10, 20,26], 'scorer_name':'Lowest_CWA', 'dist': "local-adaptive-squared-euclidean"}
    else:
        param_grid_gmlvq={'prot_choice': [np.array([3,2,1,1]),np.array([3,3,2,1]),np.array([3,2,2,1]),
            np.array([1,1,1,1]), np.array([1,1,2,1]), np.array([2,1, 2, 1])],
               'n_comp':[7,15, 20,29], 'scorer_name':'Lowest_CWA',  'dist': "adaptive-squared-euclidean"}
        param_grid_lgmlvq={'prot_choice': [np.array([3,2,1,1]),np.array([3,3,2,1]),np.array([3,2,2,1]),
        np.array([1,1,1,1]), np.array([1,1,2,1]), np.array([2,1, 2, 1])], 'n_comp':[7,15, 20,29], 
                           'scorer_name':'Lowest_CWA', 'dist': "local-adaptive-squared-euclidean"}
    GridSearch_LVQ(X,Y, sampling_strategy, param_grid_gmlvq)
    GridSearch_LVQ(X, Y, sampling_strategy, param_grid_lgmlvq)

# Dataset type 1
adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['EDEQ']))
df_train_adapted=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
YTrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
df_test_adapted=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
YTest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()
nan_mean=np.nanmean(df_train_adapted.to_numpy(), axis=0)
nan_std=np.nanstd(df_train_adapted.to_numpy(), axis=0)
z_train_explore_nan=((df_train_adapted.to_numpy()-nan_mean)/nan_std)
z_test_df=pd.DataFrame(data=((df_test_adapted.to_numpy()-nan_mean)/nan_std), columns=adapted_combo_cols)
z_train_df=pd.DataFrame(data=z_train_explore_nan, columns=adapted_combo_cols)
kernel_mean_match = mf.ImputationKernel(data=z_train_df,num_datasets=1,mean_match_candidates=5)
kernel_mean_match.mice(10)
z_train_explore=pd.DataFrame(data=kernel_mean_match.complete_data(), columns=adapted_combo_cols)
temp_test=kernel_mean_match.impute_new_data(z_test_df)
z_test_explore=pd.DataFrame(data=temp_test.complete_data(), columns=adapted_combo_cols)
print('For dataset with Core stuff only and 5 classes:\n')
LVQ_Hyperparameters(z_train_explore, YTrain)

## Dataset type 2:
df_train_eds=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & (df_adapted_combo['EDtype']!='Others')]
YTrain_eds=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') & (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
df_test_eds=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & (df_adapted_combo['EDtype']!='Others')]
YTest_eds=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') & (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
nan_mean=np.nanmean(df_train_eds.to_numpy(), axis=0)
nan_std=np.nanstd(df_train_eds.to_numpy(), axis=0)
z_train_eds_nan=((df_train_eds.to_numpy()-nan_mean)/nan_std)
z_train_ed_df=pd.DataFrame(data=z_train_eds_nan, columns=adapted_combo_cols)
z_test_ed_df=pd.DataFrame(data=((df_test_eds.to_numpy()-nan_mean)/nan_std), columns=adapted_combo_cols)

kernel_mm= mf.ImputationKernel(data=z_train_ed_df,num_datasets=1,mean_match_candidates=5)
kernel_mm.mice(10)
z_train_eds=pd.DataFrame(data=kernel_mm.complete_data(), columns=adapted_combo_cols)
temp_test=kernel_mm.impute_new_data(z_test_ed_df)
z_test_eds=pd.DataFrame(data=temp_test.complete_data(), columns=adapted_combo_cols)
print('For dataset with Core stuff only and only ED classes:\n')
LVQ_Hyperparameters(z_train_eds, YTrain_eds)
