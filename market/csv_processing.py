import pandas as pd
#basic
import numpy as np
#viz
import pickle
#others
import subprocess
from subprocess import check_output
import gc


#train = pd.read_csv('./input/train.csv')
# transactions = pd.read_csv('./input/transactions.csv')
items = pd.read_csv('./input/items.csv')
items.to_pickle("randomtest.pkl")
# holidays = pd.read_csv('./input/holidays_events.csv')
# stores = pd.read_csv('./input/stores.csv')
# oil = pd.read_csv('./input/oil.csv')
# test = pd.read_csv('./input/test.csv')

#print(train.head())
# print("="*20)
# print(transactions.head())
# print("="*20)
# print(items.head())
# print("="*20)
# print(holidays.head())
# print("="*20)
# print(stores.head())
# print("="*20)
# print(oil.head())
# print("="*20)
# print(test.head())

# Reduce memory of test

# mem_test=test.memory_usage(index=True).sum()
# #print("train dataset uses ",mem_train/ 1024**2," MB")
# print("test dataset uses ",mem_test/ 1024**2," MB")
#
# #There are only 54 stores
# test['store_nbr'] = test['store_nbr'].astype(np.uint8)
# # The ID column is a continuous number from 1 to 128867502 in train and 128867503 to 125497040 in test
# test['id'] = test['id'].astype(np.uint32)
# # item number is unsigned
# test['item_nbr'] = test['item_nbr'].astype(np.uint32)
# #Converting the date column to date format
# test['date']=pd.to_datetime(test['date'],format="%Y-%m-%d")
# #check memory
# print(test.memory_usage(index=True))
# new_mem_test=test.memory_usage(index=True).sum()
# print("test dataset uses ",new_mem_test/ 1024**2," MB after changes")
# print("memory saved =",(mem_test-new_mem_test)/ 1024**2," MB")
# test[['id','date', 'store_nbr', 'item_nbr', 'onpromotion']].to_csv('new_test.csv', index=False)

# Reduce memory of train

# now scaling it to the entire dataset of train

# scaling part 1 to the entire dataset
dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32,
            "unit_sales":np.float32
           }

train_part2=pd.read_csv("./input/train.csv",dtype=dtype_dict,usecols=[1,5],parse_dates=[0])
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



train_part1 = pd.read_csv("./input/train.csv",dtype=dtype_dict,usecols=[0,2,3,4])
print(train_part1.dtypes)

# joining part one and two
# For people familiar with R , the equivalent of cbind in pandas is the following command
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
