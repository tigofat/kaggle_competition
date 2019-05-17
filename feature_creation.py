import numpy as np
import pandas as pd

import toolbox as tb


class YNormal:
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.delta = data.std(axis=0)

    def transform(self, data):
        trans_data = (data - self.mean) / self.delta
        return trans_data


def featurize(train, test):
    """
    train: train dataframe with target column
    test: test dataframe without target column
    
    will preprocess train and test data (or train and validation data) 
    and make sure there are no NaNs,
    will delete target column from train and return it in list
    """
    used_columns = ['id', 'belongs_to_collection', 'budget',
                    'genres', 'homepage', 'imdb_id',
                    'original_language', 'original_title',
                    'overview', 'popularity', 'poster_path',
                    'production_companies', 'release_date',
                    'runtime', 'spoken_languages', 'status',
                    'tagline', 'title', 'Keywords', 'cast',
                    'crew']

    # join train-data and test-data
    data = np.vstack((train[used_columns], test[used_columns]))
    data = pd.DataFrame(data, columns=used_columns)
    features = pd.DataFrame()

    # get columns where features are given in json format 
    # and turn them into dicts.
    json_feat = ['belongs_to_collection', 'genres', 
                'spoken_languages', 'Keywords', 
                'cast', 'crew']

    data[json_feat] = data[json_feat].fillna(0)

    for name in json_feat:
        data[name] = tb.json_to_dict(data[name])

    # if film's 'belongs_to_collection' feature
    # is not nan: 'is_from_coll' feature is 1, else: 0
    features['is_from_coll'] = np.array(
        pd.notna(data['belongs_to_collection']), dtype=int)

    # if film's 'belongs_to_collection' feature
    # is nan: 'not_from_coll' feature is 1, else: 0
    features['not_from_coll'] = np.array(
        pd.isna(data['belongs_to_collection']), dtype=int)

    # normalize budget
    norm_budget = YNormal()
    norm_budget.fit(data.budget)
    features['budget'] = norm_budget.transform(data.budget)

    # normalize popularity
    norm_popularity = YNormal()
    norm_popularity.fit(data.popularity)
    features['popularity'] = norm_popularity.transform(data.popularity)

    # normalize runtime
    norm_runtime = YNormal()
    data.runtime = data.runtime.fillna(data.runtime.mean())
    norm_runtime.fit(data.runtime)
    features['runtime'] = norm_runtime.transform(data.runtime)

    # see comment for 'is_from_coll' feature, same for 'is_tagline' feat.
    features['is_tagline'] = np.array(
        pd.notna(data.tagline), dtype=int)
    # see comment for 'not_from_coll' feature, same for 'no_tagline' feat.
    features['no_tagline'] = np.array(
        pd.isna(data.tagline), dtype=int)

    # if film's 'original_language' is 'en': 'is_en' feat. is 1, else: 0
    features['is_en'] = np.array(
        data.original_language == 'en', dtype=int)
    # if film's 'original_language' is not 'en': 'not_en' feat. is 1,
    # else: 0
    features['not_en'] = np.array(
        data.original_language != 'en', dtype=int)

    # see comment for 'is_from_coll' feature, same for 'is_homepage' feat.
    features['is_homepage'] = np.array(
        pd.notna(data.homepage), dtype=int)
    # see comment for 'not_from_coll' feature, same for 'no_homepage' feat.
    features['no_homepage'] = np.array(
        pd.isna(data.homepage), dtype=int)


    """ CREATE MORE FEATURES """

    # Create features for 'genres'.
    # As we saw in notebook, 'Action', 'Adventure', 'Drama'
    # genres were the top 3 most important, 
    # so we create features, by setting 
    # all films that have one of these gnres to 1, otherwize 0
    features['popular_genre'] = 0
    features['non_popular_genre'] = 0

    for i, row in data['genres'].iteritems():

        feature_names = [feature['name'] for feature in row]

        pop, non_pop = 0, 1

        if set(feature_names) & set(['Action', 'Adventure', 
                                    'Drama']):
            pop, non_pop = non_pop, pop

        # 'iat' function sets the value to given
        features['popular_genre'].iat[i] = pop
        features['non_popular_genre'].iat[i] = non_pop


    # Do that same with 'non_pop_languages' and other columns 
    # as we did from 'genres'
    # features['pop_languages'] = 0
    # features['non_pop_languages'] = 0

    # for i, row in data['spoken_languages'].iteritems():

    #     feature_names = [feature['name'] for feature in row]

    #     pop, non_pop = 0, 1

    #     if set(feature_names) & set(['English', 'Español', 
    #                                 'Français', 'Deutsch', 
    #                                 'Italiano', 'Pусский']):
    #         pop, non_pop = non_pop, pop

    #     # 'iat' function sets the value to given
    #     features['pop_languages'].iat[i] = pop
    #     features['non_pop_languages'].iat[i] = non_pop


    # 'cast'
    features['actor'] = 0
    features['no_actor'] = 0

    for i, row in data['cast'].iteritems():

        feature_names = [feature['name'] for feature in row]

        actor, no_actor = 0, 10

        if set(feature_names) & set(['Samuel L. Jackson', 'Stan Lee', 
                                    'Frank Welker', 'Jeremy Renner', 
                                    'Johnny Depp', 'Tyrese Gibson',
                                    'Judi Dench', 'John Turturro',
                                    'John Turturro', 'Shia LaBeouf',
                                    'Scarlett Johansson', 'Jess Harnell']):
            actor, no_actor = no_actor, actor

        # 'iat' function sets the value to given
        features['actor'].iat[i] = actor
        features['no_actor'].iat[i] = no_actor

    """..."""

    targets = train[["revenue"]]
    # reseparating featurized train and test data
    train_features = features[:train.shape[0]]
    test_features = features[
                    train.shape[0]:(train.shape[0] + test.shape[0])]

    return train_features, test_features, targets
    
