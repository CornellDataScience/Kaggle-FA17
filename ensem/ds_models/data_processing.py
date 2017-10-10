

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
    train_features = []
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
                and not 'number' in c:
            cat_feature_inds.append(i)

    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
    return cat_feature_inds