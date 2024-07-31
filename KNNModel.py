import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.n_neighbors = 1
        self.classifer = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def desc(self):
        '''
        This function is to simply print out general information of the dataset
        '''
        print("Description of dataset:\n",self.dataset.DESCR[:200])
        print("Keys in the dataset:\n", self.dataset.keys())
        print("Number of data points & features:\n", self.dataset.data.shape)
        print("Feature names:\n",self.dataset.feature_names)
        print("Sample counts per class:\n",
              {n: v for n, v in zip(self.dataset.target_names, np.bincount(self.dataset.target))})

    def split(self):
        '''
        This function is to split dataset into 4 groups with 2 for training and other 2 for testing
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.dataset['data'],self.dataset['target'], random_state=0)
        return X_train, X_test, y_train, y_test

    def train(self,neighbors,features_set,target_set):
        '''
        This function is to build the model with training dataset
        '''
        if neighbors != self.n_neighbors:
            self.n_neighbors = neighbors
            self.classifer = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.classifer.fit(features_set,target_set)
        #return self.classifer

    def pred(self,dataset):
        '''
        This function is to make prections on the given dataset
        '''
        return self.classifer.predict(dataset)

    def evaluate(self,evaluating_set,true_labels):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        return round(self.classifer.score(evaluating_set,true_labels)*100,2)

