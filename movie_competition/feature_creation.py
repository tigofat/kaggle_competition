import numpy as np
import pandas as pd

from collections import Counter

import toolbox as tb


class YNormal:
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.delta = data.std(axis=0)

    def transform(self, data):
        trans_data = (data - self.mean) / self.delta
        return trans_data

    
def my_get_dummies(data, features):

    """ :param
        data:  pd.series, data which must be featurized
        features:  list, feature values for which will be dummies

        :returns
        dataframe with columns named by features list and
        with values 1 or 0 in each row of each column
    """

    filled_data = data.fillna('none')
    dataframe = []
    ary = []
    for feature in features:
        for i in range(len(data)):
            ary.append(1 if feature in filled_data[i] else 0)
        dataframe.append(ary)
        ary = []
    dataframe = pd.DataFrame(np.array(dataframe).T, columns=features)

    return  dataframe


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
#     json_feat = ['belongs_to_collection', 'genres', 
#                 'spoken_languages', 'Keywords', 
#                 'cast', 'crew']

#     data[json_feat] = data[json_feat].fillna('')

#     for name in json_feat:
#         data[name] = tb.json_to_dict(data[name])
        
#     features['genders_0_crew'] = data.crew.apply(lambda x: sum(1 for item in x if item['gender'] == 0))
#     features['genders_1_crew'] = data.crew.apply(lambda x: sum(1 for item in x if item['gender'] == 1))
#     features['genders_2_crew'] = data.crew.apply(lambda x: sum(1 for item in x if item['gender'] == 2))

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
    features['budget'] = norm_budget.transform(data.budget).fillna(0).apply(np.log1p)

    # normalize popularity
    norm_popularity = YNormal()
    norm_popularity.fit(data.popularity)
    features['popularity'] = norm_popularity.transform(data.popularity).fillna(0).apply(np.log1p)

    # normalize runtime
    norm_runtime = YNormal()
    data.runtime = data.runtime.fillna(data.runtime.mean())
    norm_runtime.fit(data.runtime)
    features['runtime'] = norm_runtime.transform(data.runtime.fillna(data.runtime.mean()).apply(np.log1p))

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
    
    # add release data feature
    fixed_years = data['release_date'].replace(np.nan, '')
    years = np.array([f'20{date[-2:]}' if date[-2:].startswith('0') else f'19{date[-2:]}' 
                  for date in fixed_years.values], dtype=int)
    
    release_date_mask = np.array((years >= 1980), dtype=np.bool)
    features['is_date'] = release_date_mask.astype(np.uint8)
    features['not_date'] = (~release_date_mask).astype(np.uint8)
    
    # if the films title is the same 1, otherwise 0
#     features['is_same_title'] = 0
#     features['not_same_title'] = 0
#     same_title_mask = data.original_title == data.title
#     features['is_same_title'][same_title_mask] = 1
#     features['not_same_title'][~same_title_mask] = 1

#     all_crew_members = [dict['name'] for dicts in train['crew'] for dict in dicts]
#     top_crew_names, _ = np.array(Counter(all_crew_members).most_common(2)).T
    
#     all_depar_members = [dict['department'] for dicts in train['crew'] for dict in dicts]
#     top_depar_names, _ = np.array(Counter(all_depar_members).most_common(2)).T
    
#     for c_name, d_name in zip(top_crew_names, top_depar_names):
#         features[f'crew_name_{c_name}'] = train['crew'].apply(lambda x: 1 if c_name in str(x) else 0)
#         features[f'depar_name_{d_name}'] = train['crew'].apply(lambda x: 1 if d_name in str(x) else 0)
        
    # reseparating featurized train and test data
    train_features = features[:train.shape[0]]
    test_features = features[
                    train.shape[0]:(train.shape[0] + test.shape[0])].reset_index()
    test_features = test_features.drop(['index'], axis=1)

    #dummie features
#     train_features = train_features.join(my_get_dummies(train['genres'],
#                                  ['Action', 'Adventure', 'Drama',
#                                   'Comedy', 'Thriller']))
#     test_features = test_features.join(my_get_dummies(test['genres'],
#                                  ['Action', 'Adventure', 'Drama',
#                                   'Comedy', 'Thriller']))

#     train_features = train_features.join(my_get_dummies(train['Keywords'],
#                                  ['duringcreditsstinger', 'aftercreditsstinger',
#                                   'superhero']))
#     test_features = test_features.join(my_get_dummies(test['Keywords'],
#                                  ['duringcreditsstinger', 'aftercreditsstinger',
#                                   'superhero']))

#     train_features = train_features.join(my_get_dummies(train['cast'],
#                                  ['Samuel L. Jackson', 'Robert De Niro', 
#                                   'Frank Welker', 'Stan Lee']))
#     test_features = test_features.join(my_get_dummies(test['cast'],
#                                  ['Samuel L. Jackson', 'Robert De Niro', 
#                                   'Frank Welker', 'Stan Lee']))
    
#     train_features = train_features.join(my_get_dummies(train['spoken_languages'],
#                                  ['English', 'Español', 'Français', 
#                                   'Deutsch']))
#     test_features = test_features.join(my_get_dummies(test['spoken_languages'],
#                                  ['English', 'Español', 'Français', 
#                                   'Deutsch']))
    
    train_features = train_features.join(my_get_dummies(train['production_companies'],
                                 ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures']))
    test_features = test_features.join(my_get_dummies(test['production_companies'],
                                 ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures']))
    
#     train_features = train_features.join(my_get_dummies(train['production_countries'],
#                                  ['United States of America', 'United Kingdom', 
#                                   'Germany', 'Canada']))
#     test_features = test_features.join(my_get_dummies(test['production_countries'],
#                                  ['United States of America', 'United Kingdom', 
#                                   'Germany', 'Canada']))
    
#     train_features = train_features.join(my_get_dummies(train['crew'],
#                                  ['Steven Spielberg']))
#     test_features = test_features.join(my_get_dummies(test['crew'],
#                                  ['Steven Spielberg']))
    
    targets = np.log1p(train[["revenue"]])

    return train_features, test_features, targets
