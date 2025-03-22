import numpy as np
import pandas as pd
import sys
import os
import pickle
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from HypOptClassifiers import lowest_cwacc
from imblearn.over_sampling import SMOTE
from GetDataReady import getDataNormalized, get_fname_exp, get_choice
from sklearn.inspection import permutation_importance
from modelTraining import load_trained_models#, get_fname_exp, get_choice
import warnings
warnings.filterwarnings('ignore')
code_path = '/'.join(os.getcwd().split('/')[0:4]) + '/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ
fname_pkl='Hyp_LVQ2.pkl'
savepicklpath='%s/pickles/'%(os.getcwd())
LVQModels='%s%s'%(savepicklpath, fname_pkl)

#save_dict={1.0: 'Core-DT-5Cls', 1.1: 'Core-DT-onlyED', 1.2: 'Core-DT-3ED-Others',
#    2.0: 'Core-DT-EDEQ-5Cls', 2.1: 'Core-DT-EDEQ-only-ED', 2.2: 'Core-DT-EDEQ-3ED-Others',
#    3.0: 'Core-5Cls', 3.1: 'Core-onlyED', 3.2: 'Core-3ED-Others'}

def model_feature_weights(keyD:float, dataset:dict):
    c, iters, nreps=0,5,10
    pkl_fname='%s/rearr_ftr_weights.pkl'%savepicklpath
    use_permutation_imp=['KNN','LDA','QDA','LSVC','RSVC']    
    dataset=getDataNormalized(keyD,0)
    if keyD>=3:        
        X=dataset['zXtrain']  
    else:
        X=dataset['mice_zXtrain']
    if os.path.exists(pkl_fname):
        with open(pkl_fname, 'rb') as f: 
            fimp_all_exps=pickle.load(f)
            fimp_all=fimp_all_exps[keyD]
    else:
        pipeClassifiers_all = load_trained_models('model_training', keyD)     
        Y=dataset['Ytrain']  
        nclasses=len(np.unique(Y))
        if keyD>=3:        
            X=dataset['zXtrain']  
        else:
            X=dataset['mice_zXtrain']
        num_features, fimp_all, permut_model=X.shape[1],{},{}
        for keyClf, clf in pipeClassifiers_all.items():             
            keyClf_temp=keyClf#.split('-')[1]
            if keyClf_temp in use_permutation_imp:               
                temp_permImp=[]
                for iter in range(iters):
                    permimp = permutation_importance(clf[iter], X, Y, n_repeats=nreps, scoring=make_scorer(lowest_cwacc))
                    permut_model[str(iter)+'-'+keyClf_temp]=permimp.importances
                    normalized_perm_imp=permut_model[str(iter)+'-'+keyClf_temp]/np.sum(permut_model[str(iter)+'-'+keyClf_temp]) 
                    temp_permImp.append(normalized_perm_imp)
                    del normalized_perm_imp, permimp       
                all_ftr_vars=np.concatenate(temp_permImp, axis=1)
                all_ftr_vars[all_ftr_vars<0.0]=0.0
                fimp_all[keyClf_temp]={'Mean': np.mean(all_ftr_vars, axis=1), 'Std':np.std(all_ftr_vars, axis=1),
                              'All': all_ftr_vars, 'cls_names': clf[0][keyClf_temp].classes_}
            else:            
                if keyClf_temp in ['GMLVQ','RF']:
                    for iter in range(iters):
                        if iter==0:
                            all_ftr_vars=np.zeros((iters,num_features))
                        if keyClf_temp=='GMLVQ':
                            all_ftr_vars[iter,:]=np.diagonal(clf[iter][keyClf_temp].lambda_)
                        else:
                            all_ftr_vars[iter,:]=clf[iter][keyClf_temp].feature_importances_.T/np.sum(clf[iter][keyClf_temp].feature_importances_.T)
                    fimp_all[keyClf_temp]={'Mean': np.mean(all_ftr_vars, axis=0), 'Std':np.std(all_ftr_vars, axis=0),
                                  'All': all_ftr_vars, 'cls_names': clf[0][keyClf_temp].classes_}
                elif keyClf_temp in ['LGMLVQ2', 'GNB', 'LogLASSO']:
                    for iter in range(iters):
                        if iter==0:
                            all_ftr_vars=np.zeros((iters, num_features, nclasses))
                        cwRel=np.zeros((nclasses, num_features))
                        for x in range(nclasses):         
                            if keyClf_temp=='LGMLVQ2':
                                cwRel[x,:]=np.diagonal(clf[iter][keyClf_temp].lambda_[x])
                            elif keyClf_temp=='LogLASSO':
                                temp_coeff=np.abs(clf[iter][keyClf_temp].coef_[x])
                                cwRel[x,:]=temp_coeff/np.sum(temp_coeff)
                            else:
                                temp_coeff=clf[iter][keyClf_temp].var_[x]
                                cwRel[x,:]=temp_coeff/np.sum(temp_coeff)  
                        all_ftr_vars[iter,:,:]=cwRel.T
                    perClass={}
                    for c in range(nclasses):
                        perClass['C%d'%(c+1)]={'Mean': np.mean(all_ftr_vars[:,:,c], axis=0),
                             'Std': np.std(all_ftr_vars[:,:,c], axis=0),'All': all_ftr_vars[:,:,c]}
                    perClass['cls_names']= clf[0].classes_
                    fimp_all[keyClf]=perClass.copy()
                    del perClass
                else:                    
                    print('%s Clf does not exist!'%keyClf)
    return fimp_all
            
def feature_imp_all_exp(dataset_types:dict):
    choice_dict=get_choice()
    pkl_fname='%s/rearr_ftr_weights.pkl'%savepicklpath
    if os.path.isfile(pkl_fname):
        with open(pkl_fname, 'rb') as f: 
            fimp_all_exps=pickle.load(f)
    else:
        fimp_all_exps={}
        for keyD, valD in choice_dict.items():
            dataset=dataset_types[keyD]
            fimp_all_exps[keyD]=model_feature_weights(keyD,dataset)
        with open(pkl_fname, 'wb') as f:  
            pickle.dump(fimp_all_exps, f)
    return fimp_all_exps
    
def prototypes(keyD:float):
    save_dict=get_fname_exp()
    c, dataset_types, iters, nreps=0,{},5,10    
    pkl_fname='%s/new_Prots_all_exp.pkl'%savepicklpath    
    if os.path.exists(pkl_fname):
        prots_all_exps=pickle.load(f)        
        prots_lvq=prots_all_exps[keyD]
    else:
        pipeClassifiers_all = load_trained_models('model_training', keyD)
        prots_lvq={}
        for keyClf, clf in pipeClassifiers_all.items():                
            if keyClf in ['GMLVQ','LGMLVQ']:
                for iter in range(iters):
                    if iter==0:
                        all_prots=np.zeros((iters, clf[iter][keyClf].prototypes_.shape[0], clf[iter][keyClf].prototypes_.shape[1]))    
                    all_prots[iter,:,:]=clf[iter][keyClf].prototypes_                
                prots_lvq[keyClf]={'Mean': np.mean(all_prots,axis=0),'Std': np.std(all_prots,axis=0),'All':all_prots,
                    'num_prot_per_cls': clf[0][keyClf].prototype_n_per_class,'cls_names': clf[0][keyClf].classes_} 
    return prots_lvq

def prototypes_all_exps():
    choice_dict=get_choice()
    pkl_fname='%s/new_Prots_all_exp.pkl'%savepicklpath
    if os.path.isfile(pkl_fname):
        with open(pkl_fname, 'rb') as f: 
            prots_all_exps=pickle.load(f)
    else: 
        prots_all_exps={}
        for keyD, valD in choice_dict.items():
            prots_all_exps[keyD]=prototypes(keyD)            
        with open(pkl_fname, 'wb') as f:  
            pickle.dump(prots_all_exps, f)            
    return prots_all_exps
