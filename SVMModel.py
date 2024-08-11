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
    def getMatrix(self, set_type):
        return self.evaluation[set_type]['Confusion Matrix']

    def getScore(self,set_type):
        return self.evaluation[set_type]['score']