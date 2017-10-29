import pandas as pd
import numpy as np
import pickle
import subprocess
from subprocess import check_output
import gc

dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32
           }

test2=pd.read_csv("./test.csv",dtype=dtype_dict,usecols=[1, 4],parse_dates=[0])
test2['Year'] = pd.DatetimeIndex(test2['date']).year
test2['Month'] = pd.DatetimeIndex(test2['date']).month
test2['Day'] =pd.DatetimeIndex(test2['date']).day.astype(np.uint8)
del(test2['date'])
test2['Day']=test2['Day'].astype(np.uint8)
test2['Month']=test2['Month'].astype(np.uint8)
test2['Year']=test2['Year'].astype(np.uint16)

#impute the missing values to be -1
test2["onpromotion"].fillna(0, inplace=True)
test2["onpromotion"]=test2["onpromotion"].astype(np.int8)

test1 = pd.read_csv("./test.csv",dtype=dtype_dict,usecols=[0,2,3])

# joining part one and two
test = pd.concat([test1.reset_index(drop=True), test2], axis=1)
#drop temp files
del(test1)
del(test2)
#Further Id is just an indicator column, hence not required for analysis
del(test['id'])

test.to_pickle('test_pickle.pkl')