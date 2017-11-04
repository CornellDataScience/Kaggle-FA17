import itertools
import pandas as pd
import numpy as np
import datetime

df = pd.read_pickle("./train_pickle.pkl")
del df["store_nbr"]
del df["item_nbr"]
del df["onpromotion"]
df2 = df[1:5000000]
df4 = df[20000000:25000000]
df3 = df[85000000:90000000].append(df2)
df3 = df3.append(df4)
del df2
del df
del df4

print("Done loading dataframes")
def create_weekday(row):
    res = datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day'])).weekday()
    #print(res)
    return res

df3['weekday'] = df3.apply(lambda row: create_weekday(row), axis=1)

def cust_mean(grp):
    grp['mean'] = grp['unit_sales'].mean()
    #print("calculated mean")
    return grp

df3 = df3.groupby(['weekday']).apply(cust_mean)

#58.70928
weekday_1 = df3.loc[df3['weekday'] == 0]
print(weekday_1.head(1)) #8.552743 #8.051834 #7.783812 # 8.175241 #8.11214 #13.817 0.967
weekday_2 = df3.loc[df3['weekday'] == 1]
print(weekday_2.head(1)) #8.080296 # 7.799426 #7.575816 #7.575746 #7.507822 #12.788 0.895
weekday_3 = df3.loc[df3['weekday'] == 2]
print(weekday_3.head(1)) #10.726105 #9.029711 #8.293953 #7.800325 #7.856365 #13.338 0.9336
weekday_4 = df3.loc[df3['weekday'] == 3]
print(weekday_4.head(1)) #9.013198 # 7.463301 #7.338865 #6.96355 #7.105431 #12.1027 0.8471
weekday_5 = df3.loc[df3['weekday'] == 4]
print(weekday_5.head(1)) #8.850871 #7.894951 #8.134737 #7.71341 #7.768584 #13.232 0.9262
weekday_6 = df3.loc[df3['weekday'] == 5]
print(weekday_6.head(1)) #11.22279 # 10.248409 10.563446 #10.118937 #9.811505 #16.71 1.1697
weekday_7 = df3.loc[df3['weekday'] == 6]
print(weekday_7.head(1)) #12.411229 # 11.434611 #10.576102 #10.779309 #10.547433 #17.96 1.2572


df3[['unit_sales','weekday']].to_csv('weekday_sales4.csv.gz', index=False, compression='gzip')
