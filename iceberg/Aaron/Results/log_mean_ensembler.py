import numpy as np
import pandas as pd

#Predict on Log Mean 2 - 9 as our 8
pred_1 = (pd.read_csv('Prediction_LM_2.csv'))["is_iceberg"]
pred_2 = (pd.read_csv('Prediction_LM_3.csv'))["is_iceberg"]
pred_3 = (pd.read_csv('Prediction_LM_4.csv'))["is_iceberg"]
pred_4 = (pd.read_csv('Prediction_LM_5.csv'))["is_iceberg"]
pred_5 = (pd.read_csv('Prediction_LM_6.csv'))["is_iceberg"]
pred_6 = (pd.read_csv('Prediction_LM_7.csv'))["is_iceberg"]
pred_7 = (pd.read_csv('Prediction_LM_8.csv'))["is_iceberg"]
pred_8 = (pd.read_csv('Prediction_LM_9.csv'))["is_iceberg"] 

ensembled_predictions = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8) / 8.0

print(ensembled_predictions)
pred_df = (pd.read_csv('Prediction_LM_2.csv'))['id'].copy()
pred_df = pd.concat([pred_df, ensembled_predictions], axis=1)
print(pred_df)
print("creating csv")
pred_df.to_csv('Log_Mean_Prediction_Ensemble.csv', index = False)