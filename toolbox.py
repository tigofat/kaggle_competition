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


def get_features(dataframe, col_name, based_on, func, **kwargs):
    
    # this function is helpful whenever you need, something like: 
    # What is the maximum amount of money spent on films where played actor X.
    
    features = dataframe[col_name].unique()
    
    feature_mask = [(dataframe[col_name] == feature, feature) for feature in features]
    means = [(feature, func(dataframe[based_on][mask], **kwargs)) for mask, feature in feature_mask]

    return sorted(means, key=lambda x: x[1], reverse=True)


def json_to_dict(feature_column):
	return feature_column.apply(
			lambda x: ast.literal_eval(x) if x else dict())


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


if __name__ == "__main__":
    # Few examples using get_features function.
    fake_df = pd.DataFrame({'films': ['movie1', 'movie2', 'movie3', 'movie4'], 
                        'actors': ['actor1', 'actor1', 'actor2', 'actor2'],
                       'revenue': [100, 200, 300, 400]})
    
    # Get movie names where the revenues are the highest (sorted from best to worst).
    highest_rev_movies = get_features(fake_df, 'films', 'revenue', np.max)
    
    # Get the most expansive movie for every actor, who played in it.
    # In this case actor2 played in two films (in movie3 and movie4), but the revenue of movie3 is higher, and the same happend 
    # for actor1 and for every actor in the column as shown below.
    exp_movie_for_every_actor = get_features(fake_df, 'actors', 'revenue', np.max)
