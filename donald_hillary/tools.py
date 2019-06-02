import numpy as np
from sklearn.metrics import mean_squared_error

import ast


def count_column_cls(col, tweet_indices):
    top_values = col.value_counts()[:10]

    trump_hillary_values = [0, 0]

    for i, ids in enumerate(tweet_indices):
        handler = col.iloc[ids]
        for value in handler:
            if value in top_values:
                trump_hillary_values[i] += 1

    return trump_hillary_values
    

def json_to_dict(column):
	return [ast.literal_eval(datum) 
			for i, datum in column.iteritems() if datum]


def accuracy_score(target, predicted):

	"""Measure accuracy score.

	Parameters

	target (numpy.ndarray) : actual y values.
	predicted (numpy.ndarray) : prediced y values.

	Returns

	accuracy score (float)
	"""

	return np.sum(target == predicted) / len(target)
