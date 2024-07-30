import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

dataset = load_breast_cancer()

def desc(data=dataset):
    print("Description of dataset:\n",data.DESCR)
    print("Keys in the dataset:\n", data.keys())
    print("Number of data points & features:\n", data.data.shape)
    print("Feature names:\n",data.feature_names)
    print("Sample counts per class:\n",{n: v for n, v in zip(data.target_names, np.bincount(data.target))})

def split(data=dataset):
    X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state=0)
    return X_train, X_test, y_train, y_test

def train(neighbors,features_set,target_set):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(features_set,target_set)
    return clf

def pred(dataset,model=train):
    #testing_set = dataset[1]
    return model.predict(dataset)

def evaluate(features_test,target_test,model=train):
    return model.score(features_test,target_test)

