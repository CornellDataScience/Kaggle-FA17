import pandas as pd
import numpy as np
from fbprophet import Prophet


df = pd.read_csv('./date_and_sales.csv')
df.columns = [
  'ds',
  'y'
]
# df['y'] = np.log(df['y'])

print(df.head())

m = Prophet()
m.fit(df)


future = m.make_future_dataframe(periods=14)
future.tail(20)
