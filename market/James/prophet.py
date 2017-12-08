import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot as plt


df = pd.read_csv('./date_and_sales.csv')
df.columns = [
  'ds',
  'y'
]
# df['y'] = np.log(df['y'])

print(df.head())

m = Prophet()
m.fit(df)


future = m.make_future_dataframe(periods=17)
print("wtf")
print(future.tail())

forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

m.plot(forecast);
forecast[['ds', 'yhat']].to_csv('prophet_predictions.csv')
plt.show()
