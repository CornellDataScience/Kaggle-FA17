import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



# Load the data

print("Loading properties data....")
props = pd.read_csv("properties_2016.csv")
properties = pd.DataFrame(props)
properties.head()

properties.describe(include = "all")
properties.columns
properties.shape


print("Loading train data....")
train = pd.read_csv("train_2016_v2.csv")
train.head()
train.shape
train.describe()
train.isnull().sum()


print("Loading sample data....")
sample = pd.read_csv("sample_submission.csv")
sample.head()


#Look at the NA values in the properties data set

def get_na_sum(df):
    
    total = {}
    for i in df.columns:
        total[i] = df[i].isnull().sum()
        
    return total


df_na = get_na_sum(properties)
sns.set_style("whitegrid")
sns.barplot([100*(df_na.get(i)/df.shape[0]) for i in df_na], [i for i in df_na])
sns.despine(right=True, top=True)
fig = plt.gcf()
fig.set_size_inches(10,12)
ax = plt.gca()
ax.set_xlabel("Percentage of NA values")
ax.set_ylabel("Variables")
ax.set_title("Variables vs. Percentage of NA values in Properties dataset")
plt.show()



#Lets handle the NA values

for i in df_na:
    value = (df_na.get(i)/properties.shape[0])
    if value <= 0.85:
        fill_val = np.nanmedian(properties[i])
        properties.fillna(value= fill_val, inplace = True, axis = 1) 
        #print (xx.values)
        #properties[i] = properties[i].values
    else:
        properties[i + "_missing"] = np.where(properties[i].isnull(), 0, 1)
        properties.drop(i, inplace = True, axis = 1)


properties.dtypes


#Merge train and properties

df = train.merge(properties, how= "inner", on="parcelid")
y_train = df["logerror"]
df.drop(["logerror", "transactiondate", "parcelid"], inplace = True, axis =1)



#Define the classifier

def classifier(clf, x_train, y_train, xtest, performCV = False, cv_folds = 5, early_stopping_rounds = 50):
    
    if performCV:
        
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train.values, label=y_train.values)
        dtest = xgb.DMatrix(xtest)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=500, nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        clf.set_params(n_estimators=cvresult.shape[0])
    
    clf.fit(x_train, y_train, eval_metric = "auc")

    #Predict training set:
    train_predictions = clf.predict(x_train)
    train_predprob = clf.predict_proba(x_train)[:,1]
      
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    y_pred=[]

    for i,predict in enumerate(xgb_pred2):
        y_pred.append(str(round(predict,4)))
    y_pred=np.array(y_pred)
    
    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
    # set col 'ParceID' to first col
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    from datetime import datetime
    output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



x_test = properties.drop("parcelid", axis = 1)
df = train.merge(properties, how= "inner", on="parcelid")
y_train = df["logerror"]
df.drop(["logerror", "transactiondate", "parcelid"], inplace = True, axis =1)



#Run Xgboost
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


# xgboost params
y_mean = y_train.mean()

xgb1 = XGBClassifier(learning_rate =  0.0035,
    max_depth = 6,
    subsample = 0.80,
    objective = 'reg:linear',
    colsample_bytree = 0.9,
    reg_lambda = 0.8,   
    reg_alpha = 0.4, 
    base_score = y_mean,
    nthread = 4,
    silent = 1,
    seed = 143,
    scale_pos_weight = 1)


classifier(clf = xgb1, x_train= df, y_train = y_train, xtest = x_test, performCV = True)

