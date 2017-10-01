import pandas as pd
import cPickle as pickle



def to_pickle(df, filename, protocol=-1):
	with open (filename, 'wb') as handle:
		pickle.dump(df, handle, -1)


def from_pickle(df, filename):
	with open (filename, 'rb') as handle:
		return pickle.load(handle)


def read_csv(filename):
	return pd.read_csv(filename)




