import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
print(os.getcwd())
import pickle
savepicklpath='%s/pickles/'%(os.getcwd())
from modelTraining import load_trained_models, get_fname_exp, get_choice
from GetDataReady import getDataNormalized
from typing import TypeVar, Generic, Tuple, Union, Optional
#name_mapping=pd.read_csv('/'.join(os.getcwd().split('/')[0:5])+ '/data/Map_1.csv', sep=',')

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(np.ndarray, Generic[Shape, DType]):
    """  
    Use this to type-annotate numpy arrays, e.g. 
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass

def evaluate_cmap_auc(X:pd.DataFrame,Y:Array,labs:list,keyD:float):
    iters, nreps,dataset_types=5,10,{}
    save_dict, choice_dict=get_fname_exp(), get_choice()    
    dataset_types[keyD]=getDataNormalized(keyD)
    auc_all,split_conf_all_Exps={},{}      
    split_conf_all, clf_auc={},{}     
    pipeClassifiers_all = load_trained_models('model_training', keyD)         
    for keyClf, clf in pipeClassifiers_all.items():                                
        split_cmap, split_auc_roc=np.zeros((iters, nclasses, nclasses)),np.zeros((iters,nclasses+1))
        for iter in range(iters):                                          
            labs=clf[iter].classes_
            pred_labs, pred_proba=clf[iter].predict(X), clf[iter].predict_proba(X)
            split_cmap=confusion_matrix(Y, pred_labs, normalize='true', labels=labs)            
            split_auc_roc[iter,nclasses]=roc_auc_score(Y,pred_proba, multi_class='ovr', average='weighted')
            split_auc_roc[iter,0:nclasses]=roc_auc_score(Y, pred_proba, multi_class='ovr', average=None, labels=labs)            
        split_conf_all[keyClf]=split_cmap
        clf_auc[keyClf]=split_auc_roc.copy()            
        auc_all[keyD]=clf_auc.copy()    
    return auc_all, split_conf_all_Exps

def PerfMetrics(X:pd.DataFrame,Y:Array, split:str, labs:list, keyD:float):
    """Evaluates classwise accuracies of each model from the corresponding confusion matrix
       of the dictionary conf_all_exp. The classwise AUC values are extracted from auc_all.
       The classwise performance metrices corresponding to each model is concatenated as a
       dataframe (perf_temp) and returned.
       """
    save_dict, iters=get_fname_exp()
    auc_all,split_conf_all_Exps=evaluate_cmap_auc(X,Y,labs, keyD)
 #   if split=='Test':
 #       conf_all_Exps=evaluate_cmap_auc(split)
 #   else:
 #       conf_all_Exps=evaluate_cmap_auc(split)
    conf_all=conf_all_Exps[keyD]
    col_names1, col_names2=list(map(lambda x: 'Acc-' + x, labs)), list(map(lambda x: 'AUC-' + x, labs)), 
    col_names1.append('MAA'), col_names2.append('Avg-Wgt')        
    cw_cell_all, auc_cell_all={},{}
    for keyClf, val in conf_all.items(): 
        cmap_dict,auc_dict, nclasses=conf_all[keyClf],auc_all[keyData], len(labs)
        mean_glob, tempAcc=cmap_dict.mean(axis=0), np.zeros((iters, nclasses))
        for iter in range(iters):
            tempAcc[iter,:]=np.diag(cmap_dict[iter,:,:])
        cw_std, overall_std=np.std(tempAcc, axis=0), np.std(tempAcc)
        cw_mean, overall_mean= np.diag(mean_glob), np.mean(np.diag(mean_glob))
        cwAUC_mean=np.mean(auc_all[keyData][keyClf+'-'+split], axis=0)
        cwAUC_std=np.std(auc_all[keyData][keyClf+'-'+split], axis=0)
        disp_cw_perf=np.reshape(np.stack((cw_mean, cw_std)).ravel('F'),(1,2*nclasses))
        cwAcc_cell, cwAUC_cell=[],[]
        for idx in range(nclasses):
            cwAcc_cell.append('%.3f (%.3f) '%(cw_mean[idx], cw_std[idx]))
            cwAUC_cell.append('%.3f (%.3f) '%(cwAUC_mean[idx], cwAUC_std[idx]))
        cwAUC_cell.append('%.3f (%.3f) '%(cwAUC_mean[idx+1], cwAUC_std[idx+1]))
        cwAcc_cell.append('%.3f (%.3f) '%(overall_mean, overall_std))
        cw_cell_all['%s-%s'%(keyClf, split)]=cwAcc_cell
        auc_cell_all['%s-%s'%(keyClf, split)]=cwAUC_cell
    acc_temp, auc_temp=pd.DataFrame.from_dict(cw_cell_all).T, pd.DataFrame.from_dict(auc_cell_all).T
    acc_temp.columns, auc_temp.columns=col_names1, col_names2
    acc_temp.reset_index(inplace=True), auc_temp.reset_index(inplace=True)
    acc_temp.rename(columns={"index":"Clf-Split"}, inplace=True)	
    auc_temp.rename(columns={"index":"Clf-Split"}, inplace=True)	
    perf_temp=pd.merge(acc_temp, auc_temp, on='Clf-Split')                
    return perf_temp

def PerfMetrics_Tabulate(display_tab:dict): 
    """Dictionary of confusion matrices from all classifier models from an experiment setting 
        (corresponding to a keyD), from training set and test set are passed on to PerfMetrics() separately,
        along with the labels associated with that experiment. 
        When the train and test set performance dataframes are returned from PerfMetrics(), they 
        are concatenated into a csv file with name corresponding to name of the experiment (as found in save_dict).
    """
    choice_dict=get_choice()
    show, res_data_opt=display_tab['show'], display_tab['res_data_opt']
    keyData_choice=list(choice_dict.keys())
    save_dict=get_fname_exp()
    for keyD in keyData_choice:        
        tab_fname='tabs/new_%s.csv'%(save_dict[keyD])
        if os.path.isfile(tab_fname):
            print('CSV file exists!')
            perf_all=pd.read_csv(tab_fname, sep=',', decimal='.')
        else:
            dataset_types[keyD]=getDataNormalized(keyD)
            YTrain, YTest=dataset_types[keyD]['Ytrain'], dataset_types[keyD]['Ytest']
            print(type(zXTrain))
            if keyD>=3:        
                zXTrain, zXTest=dataset_types[keyD]['zXtrain'], dataset_types[keyD]['zXtest']  
            else:
                zXTrain, zXTest=dataset_types[keyD]['mice_zXtrain'], dataset_types[keyD]['mice_zXtest']              
            perf_train=PerfMetrics(zXTrain, YTrain, np.unique(YTrain), 'Training', keyD)
            perf_test=PerfMetrics(zXTest, YTest,np.unique(YTrain), 'Test', keyD)
            perf_all=pd.concat([perf_train,perf_test])
            perf_all.to_csv(tab_fname, index=False, sep=',', decimal='.')
            print('Tabulated performance and saved as csv file.')
        if show>0:
            if keyD==res_data_opt:
                lim=display_tab['row_lim']
                if display_tab['tail_head']=='head':
                    print(perf_all.head(lim))
                else:
                    print(perf_all.tail(lim))
