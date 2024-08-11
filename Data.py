import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.df = pd.DataFrame(data=self.dataset.data,columns=self.dataset.feature_names)
        self.df_target = pd.Series(self.dataset.target)

    def desc(self):
        '''
        This function is to simply print out general information of the dataset
        '''
        print('*'*100)
        print("Description of dataset:\n",self.dataset.DESCR[:200])
        print("Keys in the dataset:\n", self.dataset.keys())
        print("Number of data points & features:\n", self.dataset.data.shape)
        print("Feature names:\n",self.dataset.feature_names)
        print("Sample counts per class:\n",
              {n: v for n, v in zip(self.dataset.target_names, np.bincount(self.dataset.target))})
        print('*'*100)
    def getLabels(self):
        return self.df_target
    
    def split(self,random=0,Stratify=None):
        X_train, X_test, y_train, y_test = train_test_split(self.df,self.df_target,stratify=Stratify,random_state=random)
        return X_train, X_test, y_train, y_test