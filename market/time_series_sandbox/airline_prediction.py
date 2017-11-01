# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('./international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()
