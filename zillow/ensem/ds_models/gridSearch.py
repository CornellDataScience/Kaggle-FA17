from mlens.ensemble import SuperLearner
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint
from mlens.preprocessing import EnsembleTransformer
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from sklearn.base import BaseEstimator
from model_super import *
from models import *
from ensembler import *
from tqdm import tqdm
from parameters import *
from data_processing import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import scale

#TODO:  BOOST UP DECISION TREE WITH ADABOOST

#Performs grid search on base models
def base_model_gridSearch(x_train, y_train):
    #Initialize Base Models
    #xgb_mod = XGBRegressor(**getXGBParams(y_train))
    lgbm_mod = LGBMRegressor(**lightGBM_params)
    rf_mod = RandomForestRegressor(criterion='mae', verbose=20, n_jobs=-1)
    nn_mod = MLPRegressor(activation='logistic', verbose=True, solver='sgd', learning_rate='adaptive')
    el_mod = ElasticNet()
    decision_tree = DecisionTreeRegressor(criterion='mae')
    adaboost = AdaBoostRegressor()

    """rf_params_dict = {
        'n_estimators': randint(50, 130),
        'max_depth': randint(2, 15),
        'min_samples_split': randint(20, 150),
        'min_samples_leaf': randint(10, 50)
    }

    random_search = RandomizedSearchCV(rf_mod, param_distributions=rf_params_dict, n_iter= 30,
                                       scoring='neg_mean_absolute_error', verbose=20, n_jobs=-1)
    random_search.fit(x_train, y_train)
    table = pd.DataFrame(random_search.cv_results_)
    table.to_html('RandomForest.html')
    table.to_csv('RandomForest.csv', index=False, header=False, sep='\t')
    print('best RF parameters: ')
    print(random_search.best_params_)
    print('best RF score: ')
    print(random_search.best_score_) """

    """
    decision_tree_params_dict = {'max_depth': randint(3, 15),
                            'min_samples_split': randint(2, 150),
                            'min_samples_leaf': randint(1, 8)
                            }
    random_search_3 = RandomizedSearchCV(decision_tree, param_distributions=decision_tree_params_dict, n_iter=30,
                                         n_jobs=-1, scoring='neg_mean_absolute_error', verbose=10)
    random_search_3.fit(x_train, y_train)
    table = pd.DataFrame(random_search_3.cv_results_)
    table.to_html('DecisionTree.html')
    table.to_csv('DecisionTree.csv', index=False, header=False, sep='\t')
    print('best DT parameters: ')
    print(random_search_3.best_params_)
    print('best DT score: ')
    print(random_search_3.best_score_) """



    """base_learners = [('lightgbm', lgbm_mod)]
                     #('NeuralNetwork', nn_mod), ('ElasticNet', el_mod), ('DecisionTree', decision_tree)]

    param_dicts = {
                   'lightgbm':
                       {'max_bin': randint(5, 100),
                        'learning_rate': uniform(0.001, 0.2),
                        'sub_feature': uniform(0.2, 0.5),
                        'bagging_fraction': uniform(0.6, 0.4),
                        'bagging_freq': randint(20, 50),
                        'num_leaves': randint(50, 700),
                        'min_hessian': uniform(0, 0.2),
                        'n_estimators': randint(300, 700)
                       },


                   #'NeuralNetwork':
                       #{'hidden_layer_sizes': randint(30, 90),
                        #'alpha': uniform(0, 10),
                        #'max_iter': randint(150, 300),
                        #'tol': uniform(0, 0.01),
                       #},
                   #'ElasticNet':
                       #{'alpha': uniform(0.3, 15),
                        #'l1_ratio': uniform(0, 1),
                        #'tol': uniform(0.0002, 0.15),
                        #'max_iter': randint(1000, 12000)},
                   #'DecisionTree':
                       #{'max_depth': randint(3, 15),
                        #'min_samples_split': randint(2, 150),
                        #'min_samples_leaf': randint(1, 8)
                       #}
                   }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    evl = Evaluator(scorer,cv=4,verbose=True)

    evl.fit(x_train.values,  # you can pass DataFrames from mlens>=0.1.3
            y_train.values,
            estimators=base_learners,
            param_dicts=param_dicts,
            n_iter=30)  # bump this up to do a larger grid search

    table = pd.DataFrame(evl.summary)
    table.to_html('LGBM.html')
    table.to_csv('LGBM.csv', index=False, header=False, sep='\t') """
