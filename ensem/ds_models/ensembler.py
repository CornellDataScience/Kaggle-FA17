

class EnsembleBase(object):

    def __init__(self, classifiers, x_feats, y_feats):

        """ classifiers is a list of lists. The zeroeth list
            is the list of first classifiers. The next list builds
            a prediction based on the predictions of the previous
            classifiers on the list.
            x_feats: features that will be trained on
            y_feats: features that will be used predictions
        """

        self.classifiers = classifiers


    def train(self):

        """ Trains the classifiers
        """

        clf_predictions = pd.DataFrame()

        for clf in classifiers[0]:
            bundle = clf.fit()
            col = clf.predict(bundle['X_train'])
            col = col.append(clf.predict(bundle['X_test']))
            clf_predictions[clf.name] = col

        for stack in classifiers[1:]:
            for clf in stack:
                clf_predictions


    def get_oof(clf, x_train, y_train, x_test):

        """ This code was taken from Anisotrophic where he
            described and implemented this cross-validation technique
        """

        ntrain = train.shape[0]
        ntest = test.shape[0]

        NFOLDS = 5 # set folds for out-of-fold prediction
        kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


    def fit(self):
        asdf

    def correlation(self):
        asfd



