import numpy as np
import pandas as pd


lead = 'Results/resnet_batchNorm_results/'
pred = pd.read_csv(lead + 'predictions.csv')

pred_1 = (pd.read_csv(lead + 'predictions.csv'))["is_iceberg"]
pred_2 = (pd.read_csv(lead + 'predictions_2.csv'))["is_iceberg"]
pred_3 = (pd.read_csv(lead + 'predictions_3.csv'))["is_iceberg"]
pred_4 = (pd.read_csv(lead + 'new_predictions_1.csv'))["is_iceberg"]
pred_5 = (pd.read_csv(lead + 'new_predictions_2.csv'))["is_iceberg"]
pred_6 = (pd.read_csv(lead + 'new_predictions_3.csv'))["is_iceberg"]
pred_7 = (pd.read_csv(lead + 'new_predictions_4.csv'))["is_iceberg"]
pred_8 = (pd.read_csv(lead + 'new_predictions_5.csv'))["is_iceberg"] 

#Take Average
ensembled_predictions = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8) / 8.0

print(ensembled_predictions)
pred_df = pred['id'].copy()
pred_df = pd.concat([pred_df, ensembled_predictions], axis=1)
print(pred_df)
print("creating csv")
pred_df.to_csv('resnet_batchNorm_8.csv', index = False)