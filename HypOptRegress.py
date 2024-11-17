import numpy as np
import pandas as pd
import sys
import os.path
import scipy as sc
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def Normalizer(X_0):
    if np.sum(np.isnan(X_0.to_numpy()))>0:
        mean_df, std_df=np.nanmean(X_0.to_numpy(), axis=0), np.nanstd(X_0.to_numpy(), axis=0)
        X=(X_0-mean_df)/std_df
    else:
        sc=StandardScaler().fit(X_0)
        X=sc.transform(X_0)
    return X

def GridSearch_RF(X, Y, scorer):
    pipe_RF_search = Pipeline(steps=[('RandomForestRegressor', RandomForestRegressor())])
    rf_param_grid = { "RandomForestRegressor__n_estimators": [100, 250, 300, 500],           
           "RandomForestRegressor__max_features": [5,7,11,15],
                    "RandomForestRegressor__criterion":['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}
    rf_grid_search = GridSearchCV(pipe_RF_search, rf_param_grid,cv=5, scoring=scorer)
    rf_grid_search.fit(X, Y)
    df_rf_gs = pd.DataFrame(rf_grid_search.cv_results_)[
    ["param_RandomForestRegressor__n_estimators", "param_RandomForestRegressor__max_features",
     "param_RandomForestRegressor__criterion","mean_test_score", ]]
    df_rf_gs = df_rf_gs.rename(columns={"param_RandomForestRegressor__n_estimators": "n_Trees",
    "param_RandomForestRegressor__criterion":'Criterion',
    "param_RandomForestRegressor__max_features": "Max_Features", "mean_test_score": "Score"})
    df_rf_gs["Abs_Score"]=df_rf_gs["Score"].abs()
    df_rf_gs.sort_values(by=["Abs_Score"], ascending=[True], inplace=True)
    return df_rf_gs

def GridSearch_KNN(X,Y, scorer):
    pipe_KNN_search = Pipeline(steps=[('KNeighborsRegressor', KNeighborsRegressor(weights='uniform'))])
    knn_param_grid = {"KNeighborsRegressor__n_neighbors" : [3,5,7],    
               "KNeighborsRegressor__metric": ['minkowski', 'cosine', 'mahalanobis', 'seuclidean']}
    knn_grid_search = GridSearchCV(pipe_KNN_search, knn_param_grid, cv=5, scoring=scorer) 
    knn_grid_search.fit(X, Y)
    df_knn_gs = pd.DataFrame(knn_grid_search.cv_results_)[
        ["param_KNeighborsRegressor__n_neighbors", "param_KNeighborsRegressor__metric",
        "mean_test_score", ]]    
    df_knn_gs = df_knn_gs.rename(columns={"param_KNeighborsRegressor__n_neighbors": "K",
    "param_KNeighborsRegressor__metric": "Dist Metric","mean_test_score": "Score"})
    df_knn_gs["Abs_Score"]=df_knn_gs["Score"].abs()
    df_knn_gs.sort_values(by=["Abs_Score"], ascending=[True], inplace=True)
    return df_knn_gs


def GridSearch_SVM(X,Y, scorer):
    svr_linear=SVR(kernel="linear")
    svr_rbf=SVR(kernel='rbf')    
    lin_param_grid = {'svr_linear__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe_lSVM_search = Pipeline([('svr_linear', svr_linear)])
    linSVM_grid_search = GridSearchCV(pipe_lSVM_search, lin_param_grid, cv=5, scoring=scorer)
    linSVM_grid_search.fit(X, Y)
    df_linSVM_gs = pd.DataFrame(linSVM_grid_search.cv_results_)[["param_svr_linear__C","mean_test_score", ]]
    df_linSVM_gs = df_linSVM_gs.rename(columns={"param_svr_linear__C": "C", "mean_test_score": "Score"})
    df_linSVM_gs["Abs_Score"]=df_linSVM_gs["Score"].abs()
    df_linSVM_gs.sort_values(by=["Abs_Score"], ascending=[True], inplace=True)
    rbf_param_grid = {'svr_rbf__C': [0.001, 0.01, 0.1, 1, 10, 100], 
                      'svr_rbf__gamma':[0.0001, 0.001, 0.01, 1, 10, 100]}
    pipe_rSVM_search = Pipeline(steps=[('svr_rbf', svr_rbf)])
    rbfSVM_grid_search = GridSearchCV(pipe_rSVM_search, rbf_param_grid,cv=5, scoring=scorer)
    rbfSVM_grid_search.fit(X, Y)
    df_rbfSVM_gs = pd.DataFrame(rbfSVM_grid_search.cv_results_)[["param_svr_rbf__C","param_svr_rbf__gamma","mean_test_score"]]
    df_rbfSVM_gs = df_rbfSVM_gs.rename(columns={"param_svr_rbf__C": "C","param_svr_rbf__gamma": "Gamma", 
                                                "mean_test_score": "Score"})
    df_rbfSVM_gs["Abs_Score"]=df_rbfSVM_gs["Score"].abs()
    df_rbfSVM_gs.sort_values(by=["Abs_Score"], ascending=[True], inplace=True)
    return df_linSVM_gs, df_rbfSVM_gs


def GridSearchRegressors(X_0, Y):    
    X=Normalizer(X_0)
    scorer='neg_root_mean_squared_log_error'
    df_knn_gs=GridSearch_KNN(X,Y, scorer)
    print('KNN: \n', df_knn_gs.head(4))
    df_rf_gs=GridSearch_RF(X,Y, scorer)
    print('RF: \n', df_rf_gs.head(4))   
    df_linSVM_gs, df_rbfSVM_gs= GridSearch_SVM(X,Y, scorer)
    print('Linear SVM: \n', df_linSVM_gs.head(4))
    print('RBF SVM: \n', df_rbfSVM_gs.head(4))