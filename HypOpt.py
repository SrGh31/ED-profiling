import numpy as np
import pandas as pd
import sys
import os.path
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
#from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
code_path='/'.join(os.getcwd().split('/')[0:4])+'/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ
from imblearn.over_sampling import SMOTE


def maa_score(y_true, y_pred):
    maa=np.trace(confusion_matrix(y_true, y_pred, normalize='true'))/len(np.unique(y_true))    
    return maa


def Normalizer(X_0):
    if np.sum(np.isnan(X_0.to_numpy()))>0:
        mean_df, std_df=np.nanmean(X_0.to_numpy(), axis=0), np.nanstd(X_0.to_numpy(), axis=0)
        X=(X_0-mean_df)/std_df
    else:
        sc=StandardScaler().fit(X_0)
        X=sc.transform(X_0)
    return X

def ClassBalancing(X_1, Y_1, strategy):
    sm=SMOTE(sampling_strategy=strategy)
    X,Y=sm.fit_resample(X_1, Y_1)
    return X,Y

def GridSearch_RF(X, Y,repeated_kfolds):
    maa=make_scorer(maa_score)
    #sm = SMOTENC(random_state=42, sampling_strategy='not majority', categorical_features='infer')
    pipe_RF_search = Pipeline(steps=[('RandomForestClassifier', RandomForestClassifier(criterion="gini"))])
    rf_param_grid = { "RandomForestClassifier__n_estimators": [100, 250, 300, 500],           
           "RandomForestClassifier__max_features"      : [5,7,11,15]}
    rf_grid_search = GridSearchCV(pipe_RF_search, rf_param_grid,cv=repeated_kfolds, scoring=maa)
    rf_grid_search.fit(X, Y)
    df_rf_gs = pd.DataFrame(rf_grid_search.cv_results_)[
    ["param_RandomForestClassifier__n_estimators", "param_RandomForestClassifier__max_features","mean_test_score", ]]
    df_rf_gs = df_rf_gs.rename(columns={"param_RandomForestClassifier__n_estimators": "n_Trees",
    "param_RandomForestClassifier__max_features": "Max_Features", "mean_test_score": "MAA"})
    df_rf_gs.sort_values(by=["MAA"], ascending=[False], inplace=True)
    return df_rf_gs

def GridSearch_KNN(X,Y,repeated_kfolds):
    maa=make_scorer(maa_score)
    pipe_KNN_search = Pipeline(steps=[('KNeighborsClassifier', KNeighborsClassifier())])
    knn_param_grid = {"KNeighborsClassifier__n_neighbors" : [3,5,7],           
               "KNeighborsClassifier__metric": ['minkowski', 'cosine', 'mahalanobis', 'seuclidean']}
    knn_grid_search = GridSearchCV(pipe_KNN_search, knn_param_grid, cv=repeated_kfolds, scoring=maa) #'balanced_accuracy')
    knn_grid_search.fit(X, Y)
    df_knn_gs = pd.DataFrame(knn_grid_search.cv_results_)[
        ["param_KNeighborsClassifier__n_neighbors", "param_KNeighborsClassifier__metric","mean_test_score", ]]    
    df_knn_gs = df_knn_gs.rename(columns={"param_KNeighborsClassifier__n_neighbors": "K",
    "param_KNeighborsClassifier__metric": "Dist Metric", "mean_test_score": "MAA"})
    df_knn_gs.sort_values(by=["MAA"], ascending=[False], inplace=True)
    return df_knn_gs

def GridSearch_LDA(X,Y,repeated_kfolds):
    maa=make_scorer(maa_score)
    pipe_LDA_search = Pipeline(steps=[('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis())])
    lda_param_grid = {"LinearDiscriminantAnalysis__solver": ['svd', 'lsqr', 'eigen']}
    lda_grid_search = GridSearchCV(pipe_LDA_search, lda_param_grid,cv=repeated_kfolds, scoring=maa) 
    lda_grid_search.fit(X, Y)
    df_lda_gs = pd.DataFrame(lda_grid_search.cv_results_)[
        ["param_LinearDiscriminantAnalysis__solver","mean_test_score", ]]
    df_lda_gs = df_lda_gs.rename(columns={
    "param_LinearDiscriminantAnalysis__solver": "Solver", "mean_test_score": "MAA"})
    df_lda_gs.sort_values(by=["MAA"], ascending=[False], inplace=True)
    return df_lda_gs

def GridSearch_SVM(X,Y,repeated_kfolds):
    maa=make_scorer(maa_score)
    svc_linear=SVC(kernel="linear", random_state=42)
    svc_rbf=SVC(kernel='rbf', random_state=42)    
    lin_param_grid = {'svc_linear__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe_lSVM_search = Pipeline([('svc_linear', svc_linear)])
    linSVM_grid_search = GridSearchCV(pipe_lSVM_search, lin_param_grid, cv=repeated_kfolds, scoring=maa)
    linSVM_grid_search.fit(X, Y)
    df_linSVM_gs = pd.DataFrame(linSVM_grid_search.cv_results_)[["param_svc_linear__C","mean_test_score", ]]
    df_linSVM_gs = df_linSVM_gs.rename(columns={"param_svc_linear__C": "C", "mean_test_score": "MAA"})
    df_linSVM_gs.sort_values(by=["MAA"], ascending=[False], inplace=True)
    rbf_param_grid = {'svc_rbf__C': [0.001, 0.01, 0.1, 1, 10, 100], 
                      'svc_rbf__gamma':[0.0001, 0.001, 0.01, 1, 10, 100]}
    pipe_rSVM_search = Pipeline(steps=[('svc_rbf', svc_rbf)])
    rbfSVM_grid_search = GridSearchCV(pipe_rSVM_search, rbf_param_grid,cv=repeated_kfolds, scoring=maa)#'balanced_accuracy')
    rbfSVM_grid_search.fit(X, Y)
    df_rbfSVM_gs = pd.DataFrame(rbfSVM_grid_search.cv_results_)[["param_svc_rbf__C","param_svc_rbf__gamma","mean_test_score"]]
    df_rbfSVM_gs = df_rbfSVM_gs.rename(columns={"param_svc_rbf__C": "C","param_svc_rbf__gamma": "Gamma", 
                                                "mean_test_score": "MAA"})
    df_rbfSVM_gs.sort_values(by=["MAA"], ascending=[False], inplace=True)
    return df_linSVM_gs, df_rbfSVM_gs

def GridSearch_LVQ(X_0,Y_0, sampling_strategy):
    maa=make_scorer(maa_score)
    if sampling_strategy!='None':
        prot_choice=[np.ones((len(np.unique(Y_0)),),dtype=np.int16), 2*np.ones((len(np.unique(Y_0)),),dtype=np.int16)]
    else:
        if len(np.unique(Y_0))==5:
            prot_choice=[np.array([3,3,3,1,1]),np.array([3,3,2,1,1]),np.array([3,2,2,1,1]),
            np.array([2,2,2,1,1]),np.array([1,1,1,1,1])]
        else:
            prot_choice=[np.array([3,3,3,1]),np.array([3,3,2,1]),np.array([3,2,2,1]),
            np.array([2,2,2,1]),np.array([1,1,1,1])]
    repeated_kfolds = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    solvers_types = [ "sgd", "wgd", "lbfgs","adam"] 
    gmlvq=GMLVQ(distance_type="adaptive-squared-euclidean")
    zX=Normalizer(X_0)
    X,Y=ClassBalancing(zX,Y_0, sampling_strategy)
    pipeline_gmlvq = Pipeline(steps=[('gmlvq', gmlvq)])
    param_grid_gmlvq = [{"gmlvq__prototype_n_per_class":prot_choice,"gmlvq__relevance_n_components":[7,11,15],
        "gmlvq__solver_type": solvers_types, "gmlvq__activation_type": ["identity", "swish"]},
       {"gmlvq__prototype_n_per_class":prot_choice,"gmlvq__relevance_n_components":[7,11,15],
        "gmlvq__activation_params":[0.001,0.01,0.1,1], "gmlvq__solver_type":solvers_types, "gmlvq__activation_type": ["sigmoid"]}]
    gmlvq_search = GridSearchCV(pipeline_gmlvq, param_grid_gmlvq, scoring=maa, 
                                cv=repeated_kfolds,return_train_score=False)
    gmlvq_search.fit(X, Y.to_numpy())
    df_gmlvq_search = pd.DataFrame(gmlvq_search.cv_results_)[["param_gmlvq__prototype_n_per_class","param_gmlvq__solver_type",
            "param_gmlvq__relevance_n_components", "param_gmlvq__activation_type","mean_test_score"]]
    df_gmlvq_search = df_gmlvq_search.rename(columns={"param_gmlvq__prototype_n_per_class": "Prot per cls",
                        "param_gmlvq__solver_type": "Solver", "param_gmlvq__relevance_n_components":'Num Component',
                        "param_gmlvq__activation_type":'Activation type', "mean_test_score": "MAA"})
    df_gmlvq_search.sort_values(by=["MAA"], ascending=[False], inplace=True)
    print('GMLVQ: \n', df_gmlvq_search.head(4))

def GridSearchClassifiers(X_0, Y_0, sampling_strategy):    
    repeated_kfolds= RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    zX=Normalizer(X_0)
    X,Y=ClassBalancing(zX,Y_0, sampling_strategy)
    df_rf_gs=GridSearch_RF(X,Y,repeated_kfolds)
    print('RF: \n', df_rf_gs.head(4))
    df_knn_gs=GridSearch_KNN(X,Y,repeated_kfolds)
    print('KNN: \n', df_knn_gs.head(4))
    df_lda_gs=GridSearch_LDA(X,Y,repeated_kfolds)
    print('LDA: \n', df_lda_gs.head(4))
    df_linSVM_gs, df_rbfSVM_gs= GridSearch_SVM(X,Y,repeated_kfolds)
    print('Linear SVM: \n', df_linSVM_gs.head(4))
    print('RBF SVM: \n', df_rbfSVM_gs.head(4))