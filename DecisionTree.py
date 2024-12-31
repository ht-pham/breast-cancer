from time import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from report import Report
rp = Report()

class Tree:
    def __init__(self):
        self.dataset = load_breast_cancer()
        self.max_depth = None
        self.classifer = DecisionTreeClassifier(random_state=0) # a classifer with unlimited consecutive questions => overfitting
        self.stats = {'max depth':self.max_depth,'train acc':0,'test acc':0}
    
    def train(self,max_depth,X_train,y_train):
        
        if max_depth != None:
            self.max_depth = max_depth
            self.classifer = DecisionTreeClassifier(max_depth=self.max_depth,random_state=0)
        else:
            self.max_depth = None
            self.classifer = DecisionTreeClassifier(random_state=0)
        
        start = time()
        self.classifer.fit(X_train,y_train)
        # Save the model to a file
        joblib.dump(self.classifer, 'dt_model.pkl')
        print("Model saved to 'dt_model.pkl'")
        
        if max_depth == None:
            max_depth = 'Infinite'
        print("* Finished training model with {}-depth within {:.8f}".format(max_depth,time()-start))

    def pred(self,dataset,set_type):
        '''
        This function is to make prections on the given dataset
        '''
        print("* Predicting... ")
        start = time()
        y_pred = self.classifer.predict(dataset)
        end = time()
        if set_type == 'train':
            print("... training set: done within {:.8f}".format(end-start))
        else:
            print("... testing set: done within {:.8f}".format(end-start))
        return y_pred

    """ def evaluate(self,score_type,actual,predicted):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        self.stats[score_type] = round(accuracy_score(actual,predicted)*100,2)
        return self.stats[score_type] """
    