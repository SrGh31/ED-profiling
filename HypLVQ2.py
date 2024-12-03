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
code_path='/'.join(os.getcwd().split('/')[0:4])+'/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ
from EDdataset_GGZ import colsTypeCast    
from HypOpt import GridSearchClassifiers, GridSearch_LVQ
from sklvq import GMLVQ, LGMLVQ
from imblearn.over_sampling import SMOTE
from GetDataReady import getDataNormalized, LVQ_Hyperparameters


savepicklpath='%s/pickles/'%(os.getcwd())
hyp_lvq_path='%s%s.pkl'%(savepicklpath, 'Hyp_LVQ2')
check_cases=np.array([2.0, 2.1, 2.2])
df_lvq_search={}
for choice in check_cases:
    data_packet=getDataNormalized(choice)
    if data_packet['zXtrain'].isnull().sum().sum()>0:
        zXtrain, Ytrain=data_packet['mice_zXtrain'], data_packet['Ytrain']
    else:
        zXtrain, Ytrain=data_packet['zXtrain'], data_packet['Ytrain']
    df_gmlvq_search, df_lgmlvq_search=LVQ_Hyperparameters(zXtrain, Ytrain)
    df_lvq_search[choice]={'GMLVQ': df_gmlvq_search, 'LGMLVQ': df_lgmlvq_search}
    with open(hyp_lvq_path, 'wb') as f:  # open a text file
        pickle.dump(df_lvq_search, f) 