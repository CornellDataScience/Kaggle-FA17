import numpy as np
import pandas as pd

#Predict on Mean 1 - 8 as our 8
"""pred_1 = (pd.read_csv('Prediction_M_1.csv'))["is_iceberg"]
pred_2 = (pd.read_csv('Prediction_M_2.csv'))["is_iceberg"]
pred_3 = (pd.read_csv('Prediction_M_3.csv'))["is_iceberg"]
pred_4 = (pd.read_csv('Prediction_M_4.csv'))["is_iceberg"]
pred_5 = (pd.read_csv('Prediction_M_5.csv'))["is_iceberg"]
pred_6 = (pd.read_csv('Prediction_M_6.csv'))["is_iceberg"]
pred_7 = (pd.read_csv('Prediction_M_7.csv'))["is_iceberg"]
pred_8 = (pd.read_csv('Prediction_M_8.csv'))["is_iceberg"] """

pred_simple = (pd.read_csv('Mean_Prediction_Ensemble_Simple.csv'))["is_iceberg"]
pred_res = (pd.read_csv('Mean_Prediction_Ensemble_Resnet.csv'))["is_iceberg"]

#ensembled_predictions = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8) / 8.0
ensembled_predictions = (pred_simple+pred_res)/2.0
print(ensembled_predictions)
pred_df = (pd.read_csv('Prediction_M_2.csv'))['id'].copy()
pred_df = pd.concat([pred_df, ensembled_predictions], axis=1)
print(pred_df)
print("creating csv")
pred_df.to_csv('Mean_Prediction_Ensemble.csv', index = False)