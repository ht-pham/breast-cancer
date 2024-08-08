import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

class SVMModel:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.kernel = 'linear'
        self.classifer = SVC(kernel=self.kernel,random_state=0)
        self.evaluation = {
            'train':{
                'Confusion Matrix':0,
                'score':0
                    },
            'test':{
                'Confusion Matrix':0,
                'score':0
                    }
        }
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

    def train(self,kernel,X_train,y_train):
        '''
        This function is to build the model with training dataset
        '''
        if kernel != self.kernel:
            self.kernel = kernel
            self.classifer = SVC(kernel=self.kernel,random_state=0)
        self.classifer.fit(X_train,y_train)
        #return self.classifer

    def pred(self,dataset):
        '''
        This function is to make prections on the given dataset
        '''
        return self.classifer.predict(dataset)

    def evaluate(self,set_type,actual,predicted):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        self.evaluation[set_type]['Confusion Matrix'] = confusion_matrix(actual,predicted)
        self.evaluation[set_type]['score'] = round(accuracy_score(actual,predicted)*100,2)
        #return self.evaluation
