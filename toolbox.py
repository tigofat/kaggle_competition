import numpy as np
import pandas as pd

import operator
import ast

from sklearn.metrics import mean_squared_log_error


def cross_validation(estimator, X, y, folds=5, **kwargs):
    
    models_and_loss = []
    
    fold_size = len(X) // folds
    
    for fold in range(folds):
        
        mask = np.ones(len(X), dtype=bool)
        mask[fold * fold_size : (fold + 1) * fold_size] = False
        
        x_train, y_train = X[mask], y[mask]
        x_test, y_test = X[~mask], y[~mask]
        
        model = estimator.fit(x_train, y_train, **kwargs)
        score = mean_squared_log_error(model.predict(x_test), y_test)
        models_and_loss.append([model, score])
    
    return sorted(models_and_loss, key=lambda x: x[1], reverse=True)[0]


def save_as_csv(id, revenue, file_name='submissions', **kwargs):
    
    """Save predicted revenues to a spesific format for kaggle competition.
    
    Parameters
    
    id : film id.
    revenue : predecoed revenues.
    
    Returns
    
    None
    """
    
    pd.DataFrame({'revenue': revenue}, index=id).to_csv(f'{file_name}.csv', **kwargs)


def root_mean_squared_log_error(predictions, targets):
    
    """Calculate root mean squared logarithmic error.
    
    Parameters
    
    predictions : predicted target values.
    targets : actual target values.
    
    Returns
    
    loss : the loss implemeted with root mean squared logarithmic error.
    sk_loss : the same loss implemeted with sklearn package.
    """
    
    sum_ = np.sum(np.square(np.log(predictions + 1) - np.log(targets + 1)))
    loss = np.sqrt(sum_ / len(targets))
    sk_loss = np.sqrt(mean_squared_log_error(targets, predictions))
    return loss, sk_loss


def json_to_dict(feature_column):
	return feature_column.apply(
			lambda x: ast.literal_eval(x) if x else dict())


def get_unique_values(column, measure_column, oper=np.sum):
    unique_features = np.unique(column)
    
    for feature in unique_features:
        indices = np.where(column == feature)[0]
        yield feature, np.sum(measure_column[indices])


def get_json_features(feature_column, key_value, estimate_with, oper=operator.add):

	"""Featurize json feature columns easier.

	Parameters

	feature_column : pandas.Series where elements are list of dicts or falsy values.
	json_key : Name of the key, based on which estimator estimates.
	estimate_with : Array/column/list where are values for each 'feature_column' 
					(for example 'revenue')

	Returns

	dict : Sorted dictionary.
	"""

	value_names = [dict[key_value] for dicts in feature_column 
								for dict in dicts]
	value_estim = {name: 0 for name in np.unique(value_names)}
    
	for features, estim_value in zip(feature_column, estimate_with):

		for feature in features:
			value_estim[feature[key_value]] = oper(value_estim[feature[key_value]],
													estim_value)

	return sorted(value_estim.items(),
					key=lambda x: x[1],
					reverse=True)
