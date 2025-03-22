import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import make_scorer, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

code_path = '/'.join(os.getcwd().split('/')[0:4]) + '/sklvq/'
sys.path.append(code_path)
from sklvq import GMLVQ, LGMLVQ

def maa_score(y_true, y_pred):
    """
    Calculates Mean Accuracy of All Classes (MAA).
    """
    maa = np.trace(confusion_matrix(y_true, y_pred, normalize='true')) / len(np.unique(y_true))
    return maa


def lowest_cwacc(y_true, y_pred):
    """
    Calculates the lowest class-wise accuracy. This means, in a multi-class classification 
    problem, different classes have different accuracy of being identified, and this picks the
    relatively lowest accuracy from the multiple accuracies. For hyperparameter selection, this 
    lowest accuracy value is compared across all hyperparameter settings, and the setting that 
    corresponds to the relative highest value is selected. (If the lowest of the classwise accuracy
    of a setting is higher than the lowest classwise accuracy across all settings, then the former 
    setting must be the better among the combinations tried).
    """
    cwacc = np.sort(np.diag(confusion_matrix(y_true, y_pred, normalize='true')))
    min_cwacc = (cwacc[0] + cwacc[1]) / 2
    return min_cwacc


def normalizer(X_0):
    """
    Normalizes the dataframe X_0, based on the training set's mean and std values.
    """
    if np.sum(np.isnan(X_0.to_numpy())) > 0:
        mean_df, std_df = np.nanmean(X_0.to_numpy(), axis=0), np.nanstd(X_0.to_numpy(), axis=0)
        X = (X_0 - mean_df) / std_df
    else:
        sc = StandardScaler().fit(X_0)
        X = sc.transform(X_0)
    return X


def class_balancing(X_1, Y_1, strategy):
    """
    Balances the classes using SMOTE. 
    """
    sm = SMOTE(sampling_strategy=strategy)
    X, Y = sm.fit_resample(X_1, Y_1)
    return X, Y


def grid_search_lrl1(X, Y, repeated_kfolds, param_grid):
    """
    Performs grid search for Logistic Regression with L1 penalty, 
    across different values of C passed on in the dictionary param_grid.
    """
    pipe_lrl1_search = Pipeline(steps=[('LogisticRegression', LogisticRegression(penalty='l1', solver='saga',
                                                                                 class_weight='balanced'))])
    lrl1_param_grid = {"LogisticRegression__C": param_grid['LRL1_C']}
    lrl1_grid_search = GridSearchCV(pipe_lrl1_search, lrl1_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    lrl1_grid_search.fit(X, Y)
    df_lrl1_gs = pd.DataFrame(lrl1_grid_search.cv_results_)[
        ["param_LogisticRegression__C", "mean_test_score"]]
    df_lrl1_gs.rename(columns={"param_LogisticRegression__C": "C", "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_lrl1_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    return df_lrl1_gs


def grid_search_rf(X, Y, repeated_kfolds, param_grid):
    """
    Performs grid search for Random Forest Classifier.
    The hyperparameter options are passed on in the dictionary param_grid.
    """
    pipe_rf_search = Pipeline(steps=[('RandomForestClassifier', RandomForestClassifier(criterion="gini",
                                                                                      class_weight='balanced_subsample', min_samples_leaf=param_grid['RF_min_leaf']))])
    rf_param_grid = {"RandomForestClassifier__n_estimators": param_grid['RF_n_Trees'],
                     "RandomForestClassifier__max_features": param_grid['RF_Max_Features']}
    rf_grid_search = GridSearchCV(pipe_rf_search, rf_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    rf_grid_search.fit(X, Y)
    df_rf_gs = pd.DataFrame(rf_grid_search.cv_results_)[
        ["param_RandomForestClassifier__n_estimators", "param_RandomForestClassifier__max_features", "mean_test_score"]]
    df_rf_gs.rename(columns={"param_RandomForestClassifier__n_estimators": "n_Trees",
                             "param_RandomForestClassifier__max_features": "Max_Features", "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_rf_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    return df_rf_gs


def grid_search_knn(X:pd.DataFrame, Y:list, repeated_kfolds:int, param_grid:dict):
    """
    Performs grid search for K-Nearest Neighbors Classifier.
    The hyperparameter options are passed on in the dictionary param_grid. 
    """
    pipe_knn_search = Pipeline(steps=[('KNeighborsClassifier', KNeighborsClassifier())])
    knn_param_grid = {"KNeighborsClassifier__n_neighbors": [3, 5, 7],
                      "KNeighborsClassifier__metric": ['minkowski', 'cosine', 'mahalanobis', 'seuclidean']}
    knn_grid_search = GridSearchCV(pipe_knn_search, knn_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    knn_grid_search.fit(X, Y)
    df_knn_gs = pd.DataFrame(knn_grid_search.cv_results_)[
        ["param_KNeighborsClassifier__n_neighbors", "param_KNeighborsClassifier__metric", "mean_test_score"]]
    df_knn_gs.rename(columns={"param_KNeighborsClassifier__n_neighbors": "K",
                              "param_KNeighborsClassifier__metric": "Dist Metric", "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_knn_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    return df_knn_gs


def grid_search_lda(X, Y, repeated_kfolds, param_grid):
    """
    Performs grid search for Linear Discriminant Analysis.
    The hyperparameter options are passed on in the dictionary param_grid. 
    """
    pipe_lda_search = Pipeline(steps=[('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis())])
    lda_param_grid = [{"LinearDiscriminantAnalysis__solver": ['svd', 'lsqr']},
                      {"LinearDiscriminantAnalysis__shrinkage": ['auto', 'None'],
                       "LinearDiscriminantAnalysis__solver": ['eigen']}]
    lda_grid_search = GridSearchCV(pipe_lda_search, lda_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    lda_grid_search.fit(X, Y)
    df_lda_gs = pd.DataFrame(lda_grid_search.cv_results_)[
        ["param_LinearDiscriminantAnalysis__solver", "mean_test_score"]]
    df_lda_gs.rename(columns={"param_LinearDiscriminantAnalysis__solver": "Solver", "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_lda_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    return df_lda_gs


def grid_search_svm(X, Y, repeated_kfolds, param_grid):
    """
    Performs grid search for Support Vector Machine.
    The hyperparameter options are passed on in the dictionary param_grid. 
    """
    svc_linear = SVC(kernel="linear")
    svc_rbf = SVC(kernel='rbf')
    
    lin_param_grid = {'svc_linear__C': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe_lsvm_search = Pipeline([('svc_linear', svc_linear)])
    lin_svm_grid_search = GridSearchCV(pipe_lsvm_search, lin_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    lin_svm_grid_search.fit(X, Y)
    df_lin_svm_gs = pd.DataFrame(lin_svm_grid_search.cv_results_)[["param_svc_linear__C", "mean_test_score"]]
    df_lin_svm_gs.rename(columns={"param_svc_linear__C": "C", "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_lin_svm_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    
    rbf_param_grid = {'svc_rbf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'svc_rbf__gamma': [0.0001, 0.001, 0.01, 1, 10, 100]}
    pipe_rsvm_search = Pipeline(steps=[('svc_rbf', svc_rbf)])
    rbf_svm_grid_search = GridSearchCV(pipe_rsvm_search, rbf_param_grid, cv=repeated_kfolds, scoring=param_grid['scorer'])
    rbf_svm_grid_search.fit(X, Y)
    df_rbf_svm_gs = pd.DataFrame(rbf_svm_grid_search.cv_results_)[["param_svc_rbf__C", "param_svc_rbf__gamma", "mean_test_score"]]
    df_rbf_svm_gs.rename(columns={"param_svc_rbf__C": "C", "param_svc_rbf__gamma": "Gamma",
                                  "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_rbf_svm_gs.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    
    return df_lin_svm_gs, df_rbf_svm_gs


def grid_search_lvq(X_0, Y_0, sampling_strategy, param_grid):
    """
    Performs grid search for LVQ (Learning Vector Quantization).
    The hyperparameter options are passed on in the dictionary param_grid.
    The sampling_strategy selects the type of balancing of classes (if required)
    is opted by the user, and this precedes the hyperparameter selection.
    """
    prot_choice = param_grid['prot_choice']
    repeated_kfolds = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    
    if param_grid['scorer_name'] == 'MAA':
        param_grid['scorer'] = make_scorer(maa_score)
    else:
        param_grid['scorer'] = make_scorer(lowest_cwacc)
    
    solvers_types = ["sgd", "wgd", "lbfgs", "adam"]
    zX = normalizer(X_0)
    X, Y = class_balancing(zX, Y_0, sampling_strategy)
    
    if param_grid['dist'] == "adaptive-squared-euclidean":
        lvq_type = 'GMLVQ'
        lvq = GMLVQ(distance_type=param_grid['dist'])
        param_grid_lvq = [
            {"lvq__prototype_n_per_class": prot_choice, "lvq__relevance_n_components": param_grid['n_comp'],
             "lvq__solver_type": solvers_types, "lvq__activation_type": ["identity", "swish"],
             "lvq__relevance_regularization": [0, 0.001, 0.01]},
            {"lvq__prototype_n_per_class": prot_choice, "lvq__relevance_n_components": param_grid['n_comp'],
             "lvq__activation_params": [0.1, 1, 2], "lvq__solver_type": solvers_types, "lvq__activation_type": ["sigmoid"],
             "lvq__relevance_regularization": [0, 0.001, 0.01]}
        ]
    else:
        lvq_type = 'LGMLVQ'
        lvq = LGMLVQ(distance_type=param_grid['dist'])
        param_grid_lvq = [
            {"lvq__prototype_n_per_class": prot_choice, "lvq__relevance_n_components": param_grid['n_comp'],
             "lvq__solver_type": solvers_types, "lvq__activation_type": ["identity", "swish"],
             "lvq__relevance_localization": ["class"]},
            {"lvq__prototype_n_per_class": prot_choice, "lvq__relevance_n_components": param_grid['n_comp'],
             "lvq__activation_params": [0.1, 1, 2], "lvq__solver_type": solvers_types,
             "lvq__relevance_localization": ["class"], "lvq__activation_type": ["sigmoid", "soft+"]}
        ]
    
    pipeline_lvq = Pipeline(steps=[('lvq', lvq)])
    lvq_search = GridSearchCV(pipeline_lvq, param_grid_lvq, scoring=param_grid['scorer'],
                              cv=repeated_kfolds, return_train_score=False)
    lvq_search.fit(X, Y.to_numpy())
    
    if param_grid['dist'] == "adaptive-squared-euclidean":
        df_lvq_search = pd.DataFrame(lvq_search.cv_results_)[
            ["param_lvq__prototype_n_per_class", "param_lvq__solver_type",
             "param_lvq__relevance_n_components", "param_lvq__activation_type",
             'param_lvq__relevance_regularization', "mean_test_score"]]
        df_lvq_search.rename(columns={"param_lvq__relevance_localization": 'Localization type',
                                      "param_lvq__relevance_regularization": 'Reg term'}, inplace=True)
    else:
        df_lvq_search = pd.DataFrame(lvq_search.cv_results_)[
            ["param_lvq__prototype_n_per_class", "param_lvq__solver_type",
             "param_lvq__relevance_n_components", "param_lvq__activation_type",
             "param_lvq__relevance_localization", "mean_test_score"]]
        df_lvq_search.rename(columns={"param_lvq__relevance_localization": 'Localization type'}, inplace=True)
    
    df_lvq_search.rename(columns={"param_lvq__prototype_n_per_class": "Prot per cls", "param_lvq__solver_type": "Solver",
                                  "param_lvq__relevance_n_components": 'Num Component', "param_lvq__activation_type": 'Activation type',
                                  "mean_test_score": param_grid['scorer_name']}, inplace=True)
    df_lvq_search.sort_values(by=param_grid['scorer_name'], ascending=False, inplace=True)
    print(lvq_type, lvq_search.best_params_)
    return df_lvq_search


def grid_search_classifiers(X_0, Y_0, sampling_strategy, param_grid):
    """
    Sends X_0 and Y_0 along with param_grid- the dictionary containing 
    hyperparameter options for different classifiers, to the functions
    specific to the hyperparameter optimization of different classifiers,   
    which perform grid search for the corresponding classifiers and return
    the selected hyperparameter settings to this function.
    """
    repeated_kfolds = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    
    if param_grid['scorer_name'] == 'MAA':
        param_grid['scorer'] = make_scorer(maa_score)
    else:
        param_grid['scorer'] = make_scorer(lowest_cwacc)
    
    if X_0.mean().abs().mean() > 0.01:
        zX = normalizer(X_0)
    else:
        zX = X_0
    
    X, Y = class_balancing(zX, Y_0, sampling_strategy)
    df_rf_gs = grid_search_rf(X, Y, repeated_kfolds, param_grid)
    print('RF: \n', df_rf_gs.head(2))
    df_lrl1_gs = grid_search_lrl1(X, Y, repeated_kfolds, param_grid)
    print('LogLASSO: \n', df_lrl1_gs.head(2))
    df_knn_gs = grid_search_knn(X, Y, repeated_kfolds, param_grid)
    print('KNN: \n', df_knn_gs.head(2))
    df_lda_gs = grid_search_lda(X, Y, repeated_kfolds, param_grid)
    print('LDA: \n', df_lda_gs.head(2))
    df_lin_svm_gs, df_rbf_svm_gs = grid_search_svm(X, Y, repeated_kfolds, param_grid)
    print('Linear SVM: \n', df_lin_svm_gs.head(2))
    print('RBF SVM: \n', df_rbf_svm_gs.head(2))
