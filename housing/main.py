from six.moves import urllib
from zlib import crc32
import os
import tarfile
import pandas as pd
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
    plt.show()
    