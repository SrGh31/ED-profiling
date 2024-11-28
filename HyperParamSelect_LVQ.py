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
def DataSetType(choice):
    df_all_combo=pd.read_csv(fileloc_data+'maskedDAIsy_MainDect_ED_SQ48_MHC_Honos_Lav.tsv', sep='\t', decimal=',')
    df_adapted_combo, colsExtracted, subscales=colsTypeCast(df_all_combo)
    if choice==1.0:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['EDEQ']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()
        #Xtrain, XTest=df_train_adapted
        print('For dataset with Core, DT, and 5 classes:\n')
    elif choice==1.1:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['EDEQ']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        print('For dataset with Core, DT and only ED classes:\n')
    elif choice==1.2:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['EDEQ']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Other ED')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]#.to_numpy()
        print('For dataset with Core, DT and 3 ED classes and Others:\n')
    elif choice==2.0:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()
        print('For dataset with Core, DT, EDEQ subscales and all 5 classes:\n')
    elif choice==2.1:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain_eds=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest_eds=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        print('For dataset with Core, DT, EDEQ subscales, and only ED classes:\n')
    elif choice==2.2:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Other ED')]#
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]#
        print('For dataset with Core, DT, EDEQ subscales and 3 ED classes and Others:\n')       
    elif choice==3.0:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['DT']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()
        print('For dataset with Core, and all 5 classes:\n')
    elif choice==3.1:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['DT']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Others')]
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        print('For dataset with Core, and only ED classes:\n')
    elif choice==3.2:
        adapted_combo_cols=np.setdiff1d(colsExtracted,
        ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score']+list(subscales['Honos'])+list(subscales['DT']))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') 
        & (df_adapted_combo['EDtype']!='Other ED')]
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]
        print('For dataset with Core, and 3 ED classes and Others:\n')
    else:
        print('Not yet defined. Please check your choices')
    
    nan_mean=np.nanmean(Xtrain.to_numpy(), axis=0)
    nan_std=np.nanstd(Xtrain.to_numpy(), axis=0)
    z_train_explore_nan=((Xtrain.to_numpy()-nan_mean)/nan_std)
    z_train_df=pd.DataFrame(data=z_train_explore_nan, columns=adapted_combo_cols)
    z_test_df=pd.DataFrame(data=((Xtest.to_numpy()-nan_mean)/nan_std), columns=adapted_combo_cols)   
    kernel_mean_match = mf.ImputationKernel(data=z_train_df,num_datasets=1,mean_match_candidates=5)
    kernel_mean_match.mice(10)
    z_train_explore=pd.DataFrame(data=kernel_mean_match.complete_data(), columns=adapted_combo_cols)
    temp_test=kernel_mean_match.impute_new_data(z_test_df)
    z_test_explore=pd.DataFrame(data=temp_test.complete_data(), columns=adapted_combo_cols)
    data_packet={'Xtrain': Xtrain, 'Xtest': Xtest, 'Ytrain': Ytrain, 'Ytest':Ytest,
        'zXtrain': z_train_df, 'zXtest':z_test_df, 'mice_zXtrain': z_train_explore,'mice_zXtest': z_test_explore}
    return data_packet    

def HyperparameterLVQ(data_packet):
    zXtrain, Ytrain=data_packet['mice_zXtrain'], data_packet['Ytrain']
    LVQ_Hyperparameters(zXtrain, Ytrain)

def getDataNormalized_Interact():
    choice_dict={1.0: 'Core-DT, 5Cls', 1.1: 'Core-DT, only ED', 1.2: 'Core-DT, 3 ED and Others',
    2.0: 'Core-DT-EDEQ subscale, 5Cls', 2.1: 'Core-DT-EDEQ subscale, only ED', 2.2: 'Core-DT-EDEQ subscale, 3 ED and Others',
    3.0: 'Core, 5Cls', 3.1: 'Core, only ED', 3.2: 'Core, 3 ED and Others'}
    print(choice_dict)
    choice=float(input('Please enter you choice: '))
    data_packet=DataSetType(choice)
    return data_packet
        
def getDataNormalized(choice):
    data_packet=DataSetType(choice)
    return data_packet


check_cases=np.array([1.2, 2.0,2.1,2.2, 3.0, 3.1, 3.2])
for choice in check_cases:
    data_packet=getDataNormalized(choice)
    HyperparameterLVQ(data_packet)