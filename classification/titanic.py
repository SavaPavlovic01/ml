import pandas as pd
from pandas.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


data_folder = "data"
test_path =  os.path.join(data_folder, "test.csv")
train_path = os.path.join(data_folder, "train.csv")

def open_file(path = train_path):
    return pd.read_csv(path)

def split_labels_data(data):
    labels = data["Survived"].copy()
    real_data = data.drop("Survived", axis = 1)
    return real_data, labels

class Remover(BaseEstimator, TransformerMixin):
    def __init__(self, drop_name = True, drop_id = True, drop_embarked = False, drop_cabin = True, drop_age = False,
                 drop_plcass = False, drop_sex = False, drop_SibSp = False, drop_Parch = False, drop_Ticket = True, 
                 drop_Fare = False):
        self.drop_name = drop_name
        self.drop_id = drop_id
        self.drop_embarked = drop_embarked
        self.drop_cabin = drop_cabin
        self.drop_age = drop_age
        self.drop_Pclass = drop_plcass
        self.drop_sex = drop_sex
        self.drop_SibSp = drop_SibSp
        self.drop_Parch = drop_Parch
        self.drop_Ticket = drop_Ticket
        self.drop_Fare = drop_Fare
        

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        labels = []
        if self.drop_name: labels.append("Name")
        if self.drop_age: labels.append("Age")
        if self.drop_id: labels.append("PassengerId")
        if self.drop_embarked: labels.append("Embarked")
        if self.drop_cabin: labels.append("Cabin")
        if self.drop_Fare: labels.append("Fare")
        if self.drop_Pclass: labels.append("Pclass")
        if self.drop_sex: labels.append("Sex")
        if self.drop_SibSp: labels.append("SibSp")
        if self.drop_Parch: labels.append("Parch")
        if self.drop_Ticket: labels.append("Ticket")
        return X.drop(labels, axis = 1)

def drop_not_needed(data,drop_name = True, drop_id = True, drop_embarked = False, drop_cabin = True, drop_age = False,
                 drop_pclass = False, drop_sex = False, drop_SibSp = False, drop_Parch = False, drop_Ticket = True, 
                 drop_Fare = False ):
        labels = []
        if drop_name: labels.append("Name")
        if drop_age: labels.append("Age")
        if drop_id: labels.append("PassengerId")
        if drop_embarked: labels.append("Embarked")
        if drop_cabin: labels.append("Cabin")
        if drop_Fare: labels.append("Fare")
        if drop_pclass: labels.append("Pclass")
        if drop_sex: labels.append("Sex")
        if drop_SibSp: labels.append("SibSp")
        if drop_Parch: labels.append("Parch")
        if drop_Ticket: labels.append("Ticket")
        return data.drop(labels, axis = 1)


        
# vrv cu da izbacim [Name, Cabin, Ticket]
if __name__ == "__main__":
    train_data = open_file() 
    data, labels = split_labels_data(train_data)
    #train_data.info()
    #print(train_data.describe())
    #train_data.info()
    #data.hist(bins=50)
    data = drop_not_needed(data, drop_embarked= True)
    data.info()

    pls = ColumnTransformer([
         ("kek", SimpleImputer(strategy="median"), ["Age"])
    ], remainder="passthrough")

    full_pipeline = ColumnTransformer([
        ("idk", OneHotEncoder(), [1, 2]),
        ("second", StandardScaler(), [0, 5]),
    ], remainder="passthrough")

    first_pass = pls.fit_transform(data, labels)
    first_pass = pd.DataFrame(first_pass, columns = data.columns)
    first_pass.info()
    data_preped = full_pipeline.fit_transform(first_pass, labels)
    data_preped = pd.DataFrame(data_preped)
    data_preped.info()
    print(data_preped.head())
    model = RandomForestClassifier()

    
    #model.fit(data_preped, labels)
    y_train_pred = cross_val_predict(model, data_preped, labels, cv=3)
    print(cross_val_score(model, data_preped, labels, cv=3))
    print(recall_score(labels, y_train_pred))
    print(precision_score(labels, y_train_pred))
    
# KNeighbors 
# ------------------------------------
# [0.78451178 0.81818182 0.81144781]
# 0.7105263157894737
# 0.7641509433962265
    
# SGD
# ------------------------------------    
# [0.73400673 0.80808081 0.72727273]
# 0.564327485380117
# 0.7394636015325671

# Random forest
# -----------------------------------    
# [0.78787879 0.85185185 0.82491582]
# 0.7309941520467836
# 0.764525993883792
    
# SVC
# ----------------------------------
# [0.81144781 0.83164983 0.82491582]
# 0.7105263157894737
# 0.804635761589404