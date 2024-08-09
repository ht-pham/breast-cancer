from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

class Tree:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.max_depth = None
        self.classifer = DecisionTreeClassifier(random_state=0) # a classifer with unlimited consecutive questions => overfitting
        self.stats = {'max depth':self.max_depth,'train acc':0,'test acc':0}
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
        X_train, X_test, y_train, y_test = train_test_split(self.dataset.data,self.dataset.target,
                                                            stratify= self.dataset.target,random_state=42)
        return X_train, X_test, y_train, y_test
    def train(self,max_depth,X_train,y_train):
        if max_depth != None:
            self.max_depth = max_depth
            self.classifer = DecisionTreeClassifier(max_depth=self.max_depth,random_state=0)
        elif max_depth == 0:
            self.max_depth = None
            self.classifer = DecisionTreeClassifier(random_state=0)

        self.classifer.fit(X_train,y_train)

    def pred(self,dataset):
        '''
        This function is to make prections on the given dataset
        '''
        return self.classifer.predict(dataset)

    def evaluate(self,score_type,actual,predicted):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        self.stats[score_type] = round(accuracy_score(actual,predicted)*100,2)
        return self.stats[score_type]
    

            
