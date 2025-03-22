import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
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
from GetDataReady import getDataNormalized,  get_fname_exp, get_choice
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
code_path = '/'.join(os.getcwd().split('/')[0:4]) + '/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ
fname_pkl='Hyp_LVQ2.pkl'
savepicklpath='%s/pickles'%(os.getcwd())
LVQModels='%s%s'%(savepicklpath, fname_pkl)

#save_dict={1.0: 'Core-DT-5Cls', 1.1: 'Core-DT-onlyED', 1.2: 'Core-DT-3ED-Others',
#    2.0: 'Core-DT-EDEQ-5Cls', 2.1: 'Core-DT-EDEQ-only-ED', 2.2: 'Core-DT-EDEQ-3ED-Others',
#    3.0: 'Core-5Cls', 3.1: 'Core-onlyED', 3.2: 'Core-3ED-Others'}

    
def model_param_init(c:int, keyD, iter:int):
    """
    This modules initializes all model parameters for the different 
    datasets selected.
    """
    save_dict=get_fname_exp()
    nTrees, max_ftrs=np.array([500,500,100, 300,500,500, 500,300,300]), np.array([5,5,7, 10,7,5, 7,5,7])
    C_LR,knn=np.array([1,1,1, 1,1,1, 1,1,1]), np.array([3,3,3, 3,3,3, 3,3,3])
    knn_metric=['minkowski','cosine','cosine', 'cosine','cosine','cosine', 'cosine','cosine','cosine']
    lda_solver=['svd','eigen','svd', 'svd','svd','eigen',  'svd', 'svd','svd']
    C_Lin, C_RBF=np.array([100,100,1, 10,100,1, 100,10,100]),np.array([100,100,100, 100,100,100, 10, 100, 10])
    svd_gamma=np.array([0.01,0.01, 0.01, 0.01, 0.01, 0.01, 1, 0.01, 1])
    s_glob={1.0:'adam',1.1:'lbfgs',1.2:'adam', 2.0:'lbfgs',2.1:'lbfgs',2.2:'lbfgs',3.0:'lbfgs',3.1:'lbfgs',3.2:'lbfgs'}
    s_loc1={1.0:'lbfgs',1.1:'lbfgs',1.2:'lbfgs',2.0:'lbfgs',2.1:'lbfgs',2.2: 'lbfgs', 3.0:'lbfgs',3.1:'adam',3.2:'lbfgs'}
    s_loc2={1.0:'lbfgs',1.1:'lbfgs',1.2:'sgd',2.0:'lbfgs',2.1:'lbfgs',2.2: 'lbfgs', 3.0:'lbfgs',3.1:'adam',3.2:'lbfgs'}
    act_g={1.0:'swish', 1.1:'swish',1.2:'swish',2.0:'swish',2.1:'identity', 2.2:'swish',
       3.0:'swish',3.1:'swish',3.2:'swish'}
    act_l1={1.0:'swish',1.1:'swish', 1.2:'identity', 2.0:'identity',2.1:'swish',2.2:'identity',
        3.0:'identity',3.1:'swish',3.2: 'identity' }
    act_l2={1.0:'swish',1.1:'identity', 1.2:'swish', 2.0:'identity',2.1:'swish',2.2:'swish',
        3.0:'swish',3.1:'swish',3.2: 'identity' }
    rel_comp_gmlvq={1.0:'20', 1.1:'7', 1.2:'15', 2.0:'5', 2.1:'20', 2.2:'15', 3.0:'10', 3.1:'7', 3.2:'15'}
    rel_comp_lgmlvq1={1.0:'10', 1.1:'15', 1.2:'7', 2.0:'20', 2.1:'7', 2.2:'20', 3.0:'5', 3.1:'15', 3.2:'7'}
    rel_comp_lgmlvq2={1.0:'10', 1.1:'20', 1.2:'20', 2.0:'10', 2.1:'7', 2.2:'20', 3.0:'5', 3.1:'15', 3.2:'7'}
    g_num_prots_all={1.0:np.array([2, 2, 2, 1, 1]),1.1:np.array([3, 2, 1, 1]),1.2:np.array([3, 2, 1, 1]),
        2.0:np.array([3, 2, 2, 1, 1]), 2.1:np.array([3,2, 2, 1]), 2.2:np.array([3, 2, 1, 1]),
        3.0:np.array([3, 2, 2, 1, 1]), 3.1:np.array([3, 2, 1, 1]),3.2:np.array([3,3,2,1])}
    l_num_prots_all2=g_num_prots_all.copy()
    reg_comp, rel_loc=np.array([0.001,0, 0.001, 0.001,0,0.01, 0,0.01, 0.0]),'class'
    pipeRF=Pipeline(steps=[('RF', RandomForestClassifier(criterion="gini", min_samples_leaf=5,
                                        n_estimators=nTrees[c], max_features=max_ftrs[c], random_state=iter))])
    pipeKNN=Pipeline(steps=[('KNN', KNeighborsClassifier(n_neighbors=knn[c],metric=knn_metric[c]))])
    pipeLDA=Pipeline(steps=[('LDA',LinearDiscriminantAnalysis(solver=lda_solver[c], ))])
    pipeQDA=Pipeline(steps=[('QDA', QuadraticDiscriminantAnalysis(reg_param=0.01))])
    pipeLSVC=Pipeline(steps=[('LSVC', SVC(kernel="linear", C=C_Lin[c], probability=True, 
                                          random_state=iter))])
    pipeRSVC=Pipeline(steps=[('RSVC', SVC(kernel='rbf', C=C_RBF[c], gamma=svd_gamma[c], 
                                          probability=True, random_state=iter))])
    pipeGMLVQ=Pipeline(steps=[('GMLVQ',GMLVQ(distance_type='adaptive-squared-euclidean',
        solver_type=s_glob[keyD],random_state=iter,prototype_n_per_class=g_num_prots_all[keyD], 
        relevance_regularization=reg_comp[c],activation_type=act_g[keyD],
        relevance_n_components=int(rel_comp_gmlvq[keyD])))])
    pipeLGMLVQ2=Pipeline(steps=[('LGMLVQ', LGMLVQ(distance_type='local-adaptive-squared-euclidean', 
        random_state=iter,prototype_n_per_class=l_num_prots_all2[keyD], activation_type=act_l2[keyD], 
        solver_type=s_loc2[keyD],relevance_n_components=int(rel_comp_lgmlvq2[keyD]),
        relevance_localization=rel_loc))])   
    pipeGNB=Pipeline(steps=[('GNB', GaussianNB())])
    pipeLogLASSO=Pipeline(steps=[('LogLASSO',LogisticRegression(C=C_LR[c], solver='saga',
                    class_weight='balanced',penalty='l1', random_state=iter))])
    pipeClassifiers_all={'RF':pipeRF, 'KNN':pipeKNN, 'LDA':pipeLDA, 'QDA': pipeQDA, 'LSVC': pipeLSVC, 
        'RSVC': pipeRSVC, 'GMLVQ': pipeGMLVQ, 'LGMLVQ': pipeLGMLVQ2,'GNB': pipeGNB,'LogLASSO':pipeLogLASSO}
    return pipeClassifiers_all
    
def model_training():
    save_dict=get_fname_exp()
    choice_dict=get_choice()
    clf_all_exps={}
        #grid_search_classifiers(X, Y,sampling_strategy, param_grid)
    c, dataset_types, iters, nreps=0,{},5,10
    use_permutation_imp=['KNN','LDA','QDA','LSVC','RSVC']
    for keyD, val in choice_dict.items():        
        #model_fname='new_%s_clf2.pkl'%save_dict[keyD]
        modelsClassify=Path(r"%s/new_%s_clf2.pkl"%(savepicklpath, save_dict[keyD]))
        if os.path.isfile(modelsClassify):                    
            print('Model exists') #clf_all = pickle.load(open6(modelsClassify, "rb"))
            clf_all=load_trained_models('model_training', keyD)
        else:
            print('%s does not exist'%modelsClassify)
            dataset_types[keyD]=getDataNormalized(keyD,0)
            Y=dataset_types[keyD]['Ytrain']  
            if keyD>=3:        
                X=dataset_types[keyD]['zXtrain']  
            else:
                X=dataset_types[keyD]['mice_zXtrain']
            clf_all, pipe_per_clf={},{}
            for iter in range(iters): 
                pipeClassifiers_all=model_param_init(c, keyD, iter)
                for keyClf, clf in pipeClassifiers_all.items():                    
                    clf.fit(X, Y)
                    pipe_per_clf[str(iter)+'-'+keyClf]=clf                  
            for temp_clf in pipeClassifiers_all.keys():
                temp_per_clf=[]
                for tempK, tempV in pipe_per_clf.items():
                    if tempK.split('-')[1]==temp_clf:
                        temp_per_clf.append(tempV)
                clf_all[temp_clf]=temp_per_clf.copy()                
            print('Training final %s Classification models complete'%save_dict[keyD])
            with open(modelsClassify, 'wb') as f: 
                pickle.dump(clf_all, f)
        clf_all_exps[keyD]=clf_all
        del clf_all
        c+=1
    return clf_all_exps
            
def load_trained_models(model_type:str, keyD:float):
    save_dict=get_fname_exp()
    if model_type=='model_training':
        model_fname='%s_clf2.pkl'%save_dict[keyD]
        fullpath='%s/%s'%(savepicklpath, model_fname)          
       # print(fullpath)
        load_model = pickle.load(open('%s'%fullpath, "rb"))
    if model_type=='obj_features':        
        model_fname='%s/new_%s_%s.pkl'%(savepicklpath,save_dict[keyD], 'FtrImp')
        load_model=pickle.load(open(model_fname, "rb"))
    if model_type=='obj_prot':
        model_fname='%s/new_%s_%s.pkl'%(savepicklpath,save_dict[keyD], 'Prot')
        load_model=pickle.load(open(model_fname, "rb"))
    #with open(model_fname, "rb") as f:
     #   load_model=pickle.load(f)
    return load_model
