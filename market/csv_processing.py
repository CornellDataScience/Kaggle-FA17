import pandas as pd
import numpy as np
import pickle
import subprocess
from subprocess import check_output
import gc

dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32,
            "unit_sales":np.float32
           }

train_part2=pd.read_csv("./train.csv",dtype=dtype_dict,usecols=[1,5],parse_dates=[0])
train_part2['Year'] = pd.DatetimeIndex(train_part2['date']).year
train_part2['Month'] = pd.DatetimeIndex(train_part2['date']).month
train_part2['Day'] =pd.DatetimeIndex(train_part2['date']).day.astype(np.uint8)
del(train_part2['date'])
train_part2['Day']=train_part2['Day'].astype(np.uint8)
train_part2['Month']=train_part2['Month'].astype(np.uint8)
train_part2['Year']=train_part2['Year'].astype(np.uint16)

#impute the missing values to be -1
train_part2["onpromotion"].fillna(0, inplace=True)
train_part2["onpromotion"]=train_part2["onpromotion"].astype(np.int8)
print(train_part2.head())
print(train_part2.dtypes)



train_part1 = pd.read_csv("./train.csv",dtype=dtype_dict,usecols=[0,2,3,4])
print(train_part1.dtypes)

# joining part one and two
train = pd.concat([train_part1.reset_index(drop=True), train_part2], axis=1)
#drop temp files
del(train_part1)
del(train_part2)
#Further Id is just an indicator column, hence not required for analysis
id=train['id']
del(train['id'])
# check memory
print(train.memory_usage())
#The extracted train.csv file is approx 5 GB
mem_train=5*1024**3
new_mem_train=train.memory_usage().sum()
print("Train dataset uses ", new_mem_train/ 1024**2," MB after changes")
print("memory saved is approx",(mem_train-new_mem_train)/ 1024**2," MB")
train.to_pickle('train_pickle.pkl')
print("train done to pickle")
train[['Year','Month', 'Day','store_nbr','item_nbr','unit_sales', 'onpromotion']].to_csv('new_train.csv.gz', index=False, compression='gzip')



