import numpy as np
import pandas as pd


pred = pd.read_csv('Results/predictions.csv')


pred_1 = (pd.read_csv('Results/predictions.csv'))["is_iceberg"]
#pred_2 = (pd.read_csv('Results/predictions_2.csv'))["is_iceberg"]
pred_3 = (pd.read_csv('Results/predictions_3.csv'))["is_iceberg"]
pred_4 = (pd.read_csv('Results/predictions_4.csv'))["is_iceberg"]
#pred_5 = (pd.read_csv('Results/predictions_5.csv'))["is_iceberg"]
#pred_6 = (pd.read_csv('Results/predictions_6.csv'))["is_iceberg"]
#pred_7 = (pd.read_csv('Results/predictions_7.csv'))["is_iceberg"]
#pred_8 = (pd.read_csv('Results/predictions_8.csv'))["is_iceberg"] 

#Take Average
ensembled_predictions = (pred_1 + pred_3 + pred_4) / 3.0

print(ensembled_predictions)
pred_df = pred['id'].copy()
pred_df = pd.concat([pred_df, ensembled_predictions], axis=1)
print(pred_df)
print("creating csv")
pred_df.to_csv('resnet_batchNorm_3.csv', index = False)