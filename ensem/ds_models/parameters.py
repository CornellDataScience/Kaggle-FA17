


def getXGBParams(y_train):
    y_mean = y_train.mean()
    #XGBoost params
    xgb_params = {
        'n_estimators': 500,
        'eta': 0.035,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'colsample_bytree': 0.9,
        'reg_lambda': 0.8,
        'reg_alpha': 0.4,
        'base_score': y_mean,
        'eval_metric': 'mae',
        'silent': True
    }
    return xgb_params

#Refined Lightgbm params
lightGBM_params = {'max_bin': 87, 'learning_rate': 0.0033967056906684771, 'sub_feature': 0.45271612360364516,
                   'bagging_fraction': 0.86068662005424157, 'bagging_freq': 29, 'num_leaves': 143,
                   'min_hessian': 0.12543460503035603, 'n_estimators': 494, 'boosting_type': 'gbdt',
                   'objective': 'regression', 'metric': 'mae'}

#CatBoost params -- old
"""catboost_params = {
    'iterations': 200,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'verbose': False,
}"""

#Hypetuned Catboost params
catboost_params = {'bagging_temperature': 0.7218369037954099, 'border_count': 246, 'ctr_border_count': 197, 'depth': 7, 'l2_leaf_reg': 4.0, 'learning_rate': 0.030747575306233767, 'rsm': 0.8506573951709594, 'iterations': 200, 'eval_metric': 'MAE','loss_function': 'MAE'}

ridge_params = {'alpha': 2.2, 'max_iter': 6481, 'tol': 6.4782206376738009e-05}

mlp_params = {'alpha': 0.7959754829680088, 'hidden_layer_sizes': 70, 'max_iter': 430, 'tol': 0.00016583269248920019,
              'activation': 'logistic', 'solver': 'sgd', 'learning_rate': 'adaptive'}

el_params = {'alpha': 10.572547627482564, 'l1_ratio': 0.018943831556972768, 'max_iter': 2554,
             'tol': 0.069827898112191741}

# rf params
rf_params = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 100,
    'min_samples_leaf': 30,
}

#Extra Trees params
et_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 400,
    'learning_rate' : 0.75
}

#Second Layer XGBoost parameters
#previous max_depth --> 2
#previous reg_alpha --> 3.2366616300574229
#previous reg_lambda --> 2.2334612108251131
#previous subsample --> 0.94543207951317931
xgb_params_2 = {'learning_rate': 0.05962852397276576, 'subsample': 0.52,
                'reg_lambda': 0.3, 'max_depth': 3, 'reg_alpha': 0.1,
                'n_estimators': 153, 'colsample_bytree': 0.98924398017113124, 'objective': 'reg:linear',
                'eval_metric': 'mae', 'silent': 1, 'gamma': 0.36}

""""#Second Layer XGBoost parameters -- Iteration2.csv -- best Kaggle score -- 40th Percentile -- XGB, CatBoost, Lightgbm
xgb_params_2 = {'learning_rate': 0.0813787467582337, 'subsample': 0.73635809426349552,
                'reg_lambda': 3.170968556452566, 'max_depth': 2, 'reg_alpha': 3.6574070685221183, 'n_estimators': 99,
                'colsample_bytree': 0.75075140322990497, 'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': 1}
                """



"""#Second Layer XGBoost parameters -- Iteration2.csv -- 48th Percentile
xgb_params_2 = {'learning_rate': 0.041320682119474678, 'subsample': 0.7544297265725215,
                'reg_lambda': 3.7966350899885803, 'max_depth': 2, 'reg_alpha': 0.13659517451090278, 'n_estimators': 186,
                'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': 1}  """

"""Old Lightgbm parameters
#Lightgbm params
params_1 = {}
params_1['max_bin'] = 10
params_1['learning_rate'] = 0.0021 # shrinkage_rate
params_1['boosting_type'] = 'gbdt'
params_1['objective'] = 'regression'
params_1['metric'] = 'mae'          # or 'mae'
params_1['sub_feature'] = 0.345
params_1['bagging_fraction'] = 0.85 # sub_row
params_1['bagging_freq'] = 40
params_1['bagging_seed'] = 1
params_1['num_leaves'] = 512        # num_leaf
#params_1['min_data'] = 500         # min_data_in_leaf
params_1['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params_1['silent'] = False
params_1['feature_fraction_seed'] = 2
params_1['bagging_seed'] = 3
params_1['n_estimators'] = 430 """