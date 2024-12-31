from math import sqrt
from time import time
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,mean_squared_error

class KNNModel:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.n_neighbors = 1
        self.classifer = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train(self,neighbors,features_set,target_set):
        '''
        This function is to build the model with training dataset
        '''
        print("* Start building & training")
        start = time()
        if neighbors != self.n_neighbors:
            self.n_neighbors = neighbors
            self.classifer = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.classifer.fit(features_set,target_set)
        # Save the model to a file
        joblib.dump(self.classifer, 'knn_model.pkl')
        print("Model saved to 'knn_model.pkl'")
        
        end = time()
        train_time = end - start
        print("* Done building & training")
        print("* Total elapsed time: {:.8f}".format(train_time))
        #return self.classifer

    def pred(self,dataset,set_type):
        '''
        This function is to make prections on the given dataset
        '''
        if set_type =='train':
            print("* Start predicting against training set")
        else:
            print("* Start predicting against testing set")
        start = time()
        y_pred = self.classifer.predict(dataset)
        end = time()
        pred_time = end - start
        print("* Done with the current stage")
        print("* Total elapsed time: {:.8f}".format(pred_time))

        return y_pred

    def evaluate(self,true_labels,predicted_set):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        score = accuracy_score(true_labels,predicted_set)
        error = sqrt(mean_squared_error(true_labels,predicted_set))
        return round(score*100,2),error

