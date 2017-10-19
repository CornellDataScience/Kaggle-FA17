import numpy as np

# Identify columns with many missing values
def missingValueColumns(train_df):
    missing_perc_thresh = 0.98
    exclude_missing = []
    num_rows = train_df.shape[0]
    for c in train_df.columns:
        num_missing = train_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing.append(c)
    print("We exclude: %s" % exclude_missing)
    print(len(exclude_missing))

    return exclude_missing


# Identify columns with only one unique value
def nonUniqueColumns(train_df):
    exclude_unique = []
    for c in train_df.columns:
        num_uniques = len(train_df[c].unique())
        if train_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % exclude_unique)
    print(len(exclude_unique))

    return exclude_unique

#Identify columns that we will use for training
def trainingColumns(train_df, exclude_missing, exclude_unique):
    exclude_other = ['parcelid', 'logerror']  # for indexing/training only
    # do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
    exclude_other.append('propertyzoningdesc')
    train_features = ['logerror']
    for c in train_df.columns:
        if c not in exclude_missing \
                and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % train_features)
    print(len(train_features))

    return train_features

#Identify categorical columns
def categoricalColumns(train_df, train_features):
    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = len(train_df[c].unique())
        if num_uniques < cat_unique_thresh \
           and not 'sqft' in c \
           and not 'cnt' in c \
           and not 'nbr' in c \
           and not 'cos' in c \
           and not 'sin' in c \
           and not 'number' in c \
           or 'cluster' in c \
           or 'id' in c \
           or 'census' in c \
           or 'code' in c \
           or 'desc' in c:
            cat_feature_inds.append(i)
            
        
    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

    
    
    
    
"""
    
    ######## LOADING DATA #############
dir_path = path.abspath(path.join('__file__',"../../.."))
train_path = dir_path + '/train_2016_v2.csv'
train_2017_path = dir_path + '/train_2017.csv'
test_path = dir_path + '/submission.csv'
test_2017_path = dir_path + '/submission.csv'
properties_path = dir_path + '/properties_2016.csv'
properties_2017_path = dir_path + '/properties_2017.csv'
#first_layer_predictions_file = dir_path + '/predictions_first_layer.csv'

print(train_path)

train_df = pd.read_csv(train_path, parse_dates=['transactiondate'], low_memory=False)
train_2017_df = pd.read_csv(train_2017_path, parse_dates=['transactiondate'], low_memory=False)
test_df = pd.read_csv(test_path, low_memory=False)
test_2017_df = pd.read_csv(test_2017_path, low_memory=False)
properties = pd.read_csv(properties_path, low_memory=False)
properties_2017 = pd.read_csv(properties_2017_path, low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']
test_2017_df['parcelid'] = test_df['ParcelId']

########################################## PROCESSING DATA ############################################################

train_df = add_date_features(train_df)
train_2017_df = add_date_features(train_2017_df)

properties, properties_2017 = add_geographic_features(properties, properties_2017)

train_df = train_df.merge(properties, how='left', on='parcelid')
train_2017 = train_2017_df.merge(properties_2017, how='left', on='parcelid')
train_df = train_df.append(train_2017)
print("Train: ", train_df.shape)

test_df = test_df.merge(properties, how='left', on='parcelid')
test_2017 = test_2017_df.merge(properties_2017, how='left', on='parcelid')

#Identify columns with many missing values and store them into a variable
exclude_missing = missingValueColumns(train_df)

# Identify columns with only one unique value and store them into a variable
exclude_unique = nonUniqueColumns(train_df)

#Identify columns that we will use for training and store them into a variable
train_columns = trainingColumns(train_df, exclude_missing, exclude_unique)

#Identify categorical columns
cat_feature_inds = categoricalColumns(train_df, train_columns)

# Handle NA values
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)
test_2017.fillna(-1, inplace=True)

#Disregard outliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.4]

#Handle types so training does not throw errors
for c in train_df.dtypes[train_df.dtypes == object].index.values:
    train_df[c] = (train_df[c] == True)

#Initialize training datasets
train_data = train_df[train_columns]
#train_data.to_csv('training_data.csv', index = False)
print('data shape: ')
print(train_data.shape)
x_train = train_data.drop('logerror', 1)
y_train = train_data.logerror
print('train sizes: ')
print(x_train.shape)
print(y_train.shape)

#Handle types so training does not throw errors
#for c in x_train.dtypes[x_train.dtypes == object].index.values:
    #x_train[c] = (x_train[c] == True)


#Set up test dataset
test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
train_columns.remove('logerror')
X_test = test_df[train_columns]

test_2017['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_2017 = add_date_features(test_2017)
X_test_2017 = test_2017[train_columns]

#Handle types so testing does not throw errors
for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)

#Handle types so testing does not throw errors
for c in X_test_2017.dtypes[X_test_2017.dtypes == object].index.values:
    X_test_2017[c] = (X_test_2017[c] == True)

print('Test 2016 size')
print(X_test.shape)
print('Test 2017 size')
print(X_test_2017.shape)

X_test.to_csv('test_2016_data.csv', index = False)
X_test_2017.to_csv('test_2017_data.csv', index = False)  """