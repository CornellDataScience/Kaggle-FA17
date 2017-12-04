import numpy as np
import pandas as pd


lead = 'Results/'
pred = pd.read_csv(lead + 'super_ensem.csv')

pred_1 = (pd.read_csv(lead + 'super_ensem.csv'))["is_iceberg"]
pred_2 = (pd.read_csv(lead + 'stack_minmax_bestbase.csv'))["is_iceberg"]
pred_3 = (pd.read_csv(lead + 'stack_minmax_median.csv'))["is_iceberg"]

#Take Average
ensembled_predictions = (pred_1 + pred_2 + pred_3) / 3.0

print(ensembled_predictions)
pred_df = pred['id'].copy()
pred_df = pd.concat([pred_df, ensembled_predictions], axis=1)
print(pred_df)
print("creating csv")
pred_df.to_csv('stuff_of_legend_2.csv', index = False)