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
import pickle
fileloc_data='/'.join(os.getcwd().split('/')[0:5])+ '/data/annonymizedDatasets/'
savetag='pred_lavSQ_MHC'
#from EDdata_GGz import cols_type_cast
name_mapping=pd.read_csv('/'.join(os.getcwd().split('/')[0:5])+ '/data/Map_1.csv', sep=',')
# Define constants
FLOAT_DTYPES = ['float'] * 10  # Adjust the length as per requirement
# File location
#fileloc_data = os.path.join('/', *os.getcwd().split('/')[:5], 'data', 'annonymizedDatasets')

def get_fname_exp():
    """Returns the dictionary for filenames corresponding to experiments. """
    save_dict={1.0: 'Core-DT-5Cls', 1.1: 'Core-DT-onlyED', 1.2: 'Core-DT-3ED-Others',
    2.0: 'Core-DT-EDEQ-5Cls', 2.1: 'Core-DT-EDEQ-only-ED', 2.2: 'Core-DT-EDEQ-3ED-Others',
    3.0: 'Core-5Cls', 3.1: 'Core-onlyED', 3.2: 'Core-3ED-Others'}
    return save_dict
    
def get_choice():
    """Returns the dictionary mapping choice-number to experiments. """
    choice_dict={1.0: 'Core-DT, 5Cls', 1.1: 'Core-DT, only ED', 1.2: 'Core-DT, 3 ED and Others',
    2.0: 'Core-DT-EDEQ subscale, 5Cls', 2.1: 'Core-DT-EDEQ subscale, only ED', 2.2: 'Core-DT-EDEQ subscale, 3 ED and Others',
    3.0: 'Core, 5Cls', 3.1: 'Core, only ED', 3.2: 'Core, 3 ED and Others'}
    return choice_dict

def col_renaming() -> dict:
    """Return a dictionary mapping old column names to new column names."""
    cols_dectools = ['BMI', 'IND_eerdere_spec_behandeling_zonder_effect', 'aantal_eerdere_trajecten', 
                     'duur_stoornis_in_jaren', 'IND_depressie_comorbiditeit', 'IND_borderline_comorbiditeit', 
                     'IND_ocd_comorbiditeit', 'IND_anders']
    cols_edeq = ['EDEQ-eating', 'EDEQ-weight', 'EDEQ-bodyshape', 'EDEQ-lines']
    cols_honos = ['Honos-Somscore', 'Honos-Beperkingen', 'Honos-Functioneren', 'Honos-Gedragsproblemen',
                  'Honos-Symptomalogie', 'Honos-SocialeProblemen']
    cols_lav = ['Lav-Neg_Waardering', 'Lav-Gebrek_Vertrouwdheid', 'Lav-Alg_Ontevredenheid', 'Lav-Score']
    cols_sq48 = ['SQ48-Vijandigheid', 'SQ48-Agorafobie', 'SQ48-Angst', 'SQ48-Depressie', 'SQ48-Cognitieve_Klachten',
                 'SQ48-Somatische_Klachten', 'SQ48-Sociale_Fobie', 'SQ48-Vitaliteit_Optimisme', 'SQ48-Werk_Studie', 'SQ48-Score']
    cols_mhcsf = ['MHCSF-EmotionWB', 'MHCSF-SocialWB', 'MHCSF-PsychWB', 'MHCSF-Score']
    col_names = ['Main-Age', 'Main-Biosex', 'Main-Education', 'Main-ED_Codes', 'EDEQ-Score', 'EDEQ-eating', 'EDEQ-weight',
                 'EDEQ-bodyshape', 'EDEQ-lines', 'DT-BMI', 'DT-IND_prev_spec_int_wo_eff', 'DT-num_prev_routes',
                 'DT-Disorder_Duration_Yrs', 'DT-IND_depression_CMD', 'DT-IND_BDL_CMD', 'DT-IND_OCD_CMD', 'DT-IND_others',
                 'Lav-Negative_appraisal_body', 'Lav-Unfamiliarity_with_body', 'Lav-Dissatisfaction_body', 'Lav-Score',
                 'SQ48-Hostility', 'SQ48-Agoraphobia', 'SQ48-Anxiety', 'SQ48-Depression', 'SQ48-Cognitive_Complaints',
                 'SQ48-Somatic_Complaints', 'SQ48-Social_phobia', 'SQ48-Vitality', 'SQ48-Work_related_complaints', 'SQ48-Score',
                 'MHCSF-Emotional_Well-being', 'MHCSF-Social_Well-being', 'MHCSF-Psychological_Well-being', 'MHCSF-Score',
                 'Honos-Somscore', 'Honos-Limitation', 'Honos-Functionality', 'Honos-Behaviour_problem', 'Honos-Symptomalogy', 'Honos-Social_Problems']
    cols_to_consider = ['Main-Age', 'Main-Biosex', 'Main-Education', 'ED_Codes', 'EDEQ-Score'] + cols_edeq + cols_dectools + \
    cols_lav + cols_sq48 + cols_mhcsf + cols_honos
    return dict(zip(cols_to_consider, col_names))


def cols_to_float() -> dict:
    """Return a dictionary mapping column names to float types based on their substrings."""
    col_rename_dict = col_renaming()
    substrings = ['Main-Age', 'SQ48', 'MHCSF', 'Lav', 'DT', 'EDEQ', 'Honos']
    cols_renamed = {sub: [i for i in col_rename_dict.values() if sub in i] for sub in substrings}

    dtypes_to_float = {
        'DT': dict(zip(cols_renamed['DT'], FLOAT_DTYPES[:8])),
        'EDEQ': dict(zip(cols_renamed['EDEQ'], FLOAT_DTYPES[:4])),
        'Honos': dict(zip(cols_renamed['Honos'], FLOAT_DTYPES[:6])),
        'Lav': dict(zip(cols_renamed['Lav'], FLOAT_DTYPES[:4])),
        'SQ48': dict(zip(cols_renamed['SQ48'], FLOAT_DTYPES[:10])),
        'MHCSF': dict(zip(cols_renamed['MHCSF'], FLOAT_DTYPES[:4])),
    }
    return dtypes_to_float


def cols_type_cast(df: pd.DataFrame) -> (pd.DataFrame, list, dict):
    """Cast columns to appropriate data types and return the modified DataFrame along with considered columns and subscales."""
    dtypes_to_float = cols_to_float()
    col_rename_dict = col_renaming()
    df.rename(columns=col_rename_dict, inplace=True)
    main_cols = ['Main-Bsex', 'Main-Highest_Edu', 'EDtype']
    main_col_rename = ['Main-Biosex', 'Main-Education', 'ED_Codes']
    for idx, col in enumerate(main_cols):
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[main_col_rename[idx]] = df[col].cat.codes.astype(float)
    cols_to_consider, subscales = ['Main-Age', 'Main-Biosex', 'Main-Education', 'ED_Codes'],{}
    for key, dtype_dict in dtypes_to_float.items():
        res = [i for i in df.columns if key in i]
        if key != 'EDEQ':
            if res:
                for colname, dtype in dtype_dict.items():
                    df[colname] = df[colname].astype(dtype)
                cols_to_consider += list(dtype_dict.keys())
                subscales[key] = list(dtype_dict.keys())
        else:
            if len(res) > 1:
                for colname, dtype in dtype_dict.items():
                    df[colname] = df[colname].astype(dtype)
                cols_to_consider += list(dtype_dict.keys())
                subscales[key] = list(dtype_dict.keys())
            else:
                df['EDEQ-Score'] = df['EDEQ-Score'].astype(float)
                cols_to_consider.append('EDEQ-Score')
                subscales[key] = ['EDEQ-Score']
    return df[['intid', 'Split', 'EDtype'] + cols_to_consider], cols_to_consider, subscales

# Dataset type 1
def DataSetType(choice: float, show=0):
    """This function performs prepares the data based on the subscales selected (values of choice),
    ED classes selected, and imputation of missing values if applicable and returns a dictionary with
    the prepared dataset(s) and labels.
    """
    if choice<3:
        df_all_combo=pd.read_csv(fileloc_data+'maskedDAIsy_MainDect_ED_SQ48_MHC_Lav.tsv', sep='\t', decimal=',')            
    else:
        df_all_combo=pd.read_csv(fileloc_data+'maskedDAIsy_MainED_Lav_SQ48_MHCSF_Visit1.tsv', sep='\t', decimal=',')        
    df_adapted_combo, colsExtracted, subscales=cols_type_cast(df_all_combo)
    if choice<2:
        adapted_combo_cols=np.setdiff1d(colsExtracted, ['ED_Codes','EDtype','EDEQ-Score', 'SQ48-Score', 'MHCSF-Score',\
            'Lav-Score', 'Main-Biosex', 'Main-Education']+list(subscales['EDEQ']))
    elif choice<3:
        adapted_combo_cols=np.setdiff1d(colsExtracted, ['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score',\
                'EDEQ-Score', 'Main-Biosex', 'Main-Education'])
    else:
        adapted_combo_cols=np.setdiff1d(colsExtracted,['ED_Codes','EDtype', 'SQ48-Score', 'MHCSF-Score', 'Lav-Score',\
        'EDEQ-Score', 'Main-Biosex', 'Main-Education'] + list(subscales['EDEQ']))
    if choice==1.0:                
        if show>0:
            print('1.0: For dataset with Core, DT, and 5 classes (Ndims=%d):\n'%(len(adapted_combo_cols)))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()        
    elif choice==1.1:
        if show>0:
            print('1.1: For dataset with Core, DT and only ED classes:\n')        
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()        
    elif choice==1.2:
        if show>0:
            print('1.2: For dataset with Core, DT and 3 ED classes and Others:\n')
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Other ED')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]#.to_numpy()       
    elif choice==2.0:   
        if show>0:
            print('2.0: For dataset with Core, DT, EDEQ subscales and all 5 classes (Ndim=%d):\n'%(len(adapted_combo_cols)))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()        
    elif choice==2.1:
        if show>0:
            print('2.1: For dataset with Core, DT, EDEQ subscales, and only ED classes:\n')        
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Others')]#.to_numpy()
    elif choice==2.2:
        if show>0:
            print('2.2: For dataset with Core, DT, EDEQ subscales and 3 ED classes and Others:\n')    
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Other ED')]#
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]#
    elif choice==3.0:  
        if show>0:
            print('3.0: For dataset with Core only, and all 5 classes (Ndim=%d):\n'%(len(adapted_combo_cols)))
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Train']
        Ytrain=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Train']#.to_numpy()
        Xtest=df_adapted_combo[adapted_combo_cols].loc[df_adapted_combo['Split']=='Test']
        Ytest=df_adapted_combo['EDtype'].loc[df_adapted_combo['Split']=='Test']#.to_numpy()
    elif choice==3.1:
        if show>0:
            print('3.1: For dataset with Core only, and only ED classes:\n')
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') &
        (df_adapted_combo['EDtype']!='Others')]
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Others')]        
    elif choice==3.2:
        if show>0:
            print('3.2: For dataset with Core, and 3 ED classes and Others:\n')       
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other')]
        Xtrain=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Train') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytrain=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Train') 
        & (df_adapted_combo['EDtype']!='Other ED')]
        Xtest=df_adapted_combo[adapted_combo_cols].loc[(df_adapted_combo['Split']=='Test') & 
        (df_adapted_combo['EDtype']!='Other ED')]
        Ytest=df_adapted_combo['EDtype'].loc[(df_adapted_combo['Split']=='Test') &
        (df_adapted_combo['EDtype']!='Other ED')]        
    else:
        print('Not yet defined. Please check your choices')    
    Ytrain_reg=df_adapted_combo['EDEQ-Score'].loc[df_adapted_combo['Split']=='Train']
    Ytest_reg=df_adapted_combo['EDEQ-Score'].loc[df_adapted_combo['Split']=='Test']
    nan_mean=np.nanmean(Xtrain.to_numpy(), axis=0)
    nan_std=np.nanstd(Xtrain.to_numpy(), axis=0)
    z_train_explore_nan=((Xtrain.to_numpy()-nan_mean)/nan_std)
    z_train_df=pd.DataFrame(data=z_train_explore_nan, columns=adapted_combo_cols)
    z_test_df=pd.DataFrame(data=((Xtest.to_numpy()-nan_mean)/nan_std), columns=adapted_combo_cols) 
    data_packet={'Xtrain': Xtrain, 'Xtest': Xtest, 'Ytrain': Ytrain, 'Ytest':Ytest,
        'zXtrain': z_train_df, 'zXtest':z_test_df, 'Ytrain_reg': Ytrain_reg, 'Ytest_reg': Ytest_reg, 
                'mean': nan_mean, 'std': nan_std}
    if z_train_df.isnull().sum().sum()>0:
        kernel_mean_match = mf.ImputationKernel(data=z_train_df,num_datasets=1,mean_match_candidates=5)
        kernel_mean_match.mice(10)
        temp=kernel_mean_match.complete_data()
        z_train_explore=pd.DataFrame(data=temp, columns=adapted_combo_cols)
        temp_test=kernel_mean_match.impute_new_data(z_test_df)
        z_test_explore=pd.DataFrame(data=temp_test.complete_data(), columns=adapted_combo_cols)
        data_packet['mice_zXtrain'], data_packet['mice_zXtest']=z_train_explore,z_test_explore
    return data_packet

def getDataNormalized_Interact():
    choice_dict={1.0: 'Core-DT, 5Cls', 1.1: 'Core-DT, only ED', 1.2: 'Core-DT, 3 ED and Others',
    2.0: 'Core-DT-EDEQ subscale, 5Cls', 2.1: 'Core-DT-EDEQ subscale, only ED', 2.2: 'Core-DT-EDEQ subscale, 3 ED and Others',
    3.0: 'Core, 5Cls', 3.1: 'Core, only ED', 3.2: 'Core, 3 ED and Others'}
    #print(choice_dict)
    choice=float(input('Please enter you choice: '))
    data_packet=DataSetType(choice)
    return data_packet
        
def getDataNormalized(choice: float, show:int):
    """Selected dataset (based on value of choice) is normalized """
    data_packet=DataSetType(choice,show)
    return data_packet

