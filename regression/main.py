from six.moves import urllib
from sklearn.impute import SimpleImputer
from zlib import crc32
import os
import tarfile
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import model_selection
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
DATASET_PATH = "data"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = DATASET_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    tar_file = tarfile.open(tgz_path)
    tar_file.extractall(path = housing_path)
    tar_file.close()

def load_housing_data(housing_url = DATASET_PATH):
    csv_pands = os.path.join(housing_url, "housing.csv")
    return pd.read_csv(csv_pands)

def split_train_test(data, test_ration):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(id, test_ration):
    return crc32(np.int64(id)) & 0xffffffff < test_ration * 2**32

def split_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributeAdder(BaseEstimator, TransformerMixin):

    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self. household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household] 
        
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

if __name__ == "__main__":
    fetch_housing_data()
    data = load_housing_data()
    #print(data["ocean_proximity"].value_counts())
    data["income_cat"] = pd.cut(data["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1,2,3,4,5])
    
    #data.hist(bins=50, figsize=(20,15)) 
    #data["income_cat"].hist()
    #data_with_id = data.reset_index()
    #train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 42)
    split = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 42)
    
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for set_ in (strat_test_set, strat_train_set):
        set_.drop("income_cat", axis = 1, inplace = True)
    #print(data.head())
    housing = strat_train_set.drop("median_house_value", axis = 1)
    housing_labels = strat_train_set["median_house_value"].copy()
    
    #`print(housing)
    data_num = housing.drop("ocean_proximity", axis = 1)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributeAdder()),
        ("std_scaler", StandardScaler()),
    ])

    num_attribs = list(data_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    model = RandomForestRegressor()
    
    param_grid =[ {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
    ]

    housing_preped = full_pipeline.fit_transform(housing)
    grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
    grid_search.fit(housing_preped, housing_labels)
    
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis = 1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_preped = full_pipeline.transform(X_test)
    final_prediction = final_model.predict(X_test_preped)

    final_mse = mean_squared_error(y_test, final_prediction)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)