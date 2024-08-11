import csv
from statistics import mean
import numpy as np
from sklearn.metrics import accuracy_score,mean_squared_error
from math import sqrt

class Report:
    def __init__(self):
        #e.g. stats = { ml_algo:{'1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}}
        self.stats = {'knn':{},'SVM':{},'DT':{}}
        self.train_stats = {'knn':{},'SVM':{},'DT':{}}
        self.test_stats = {'knn':{},'SVM':{},'DT':{}}

        self.training_accuracy = {'knn':[],'SVM':[],'DT':[]}
        self.testing_accuracy = {'knn':[],'SVM':[],'DT':[]}
        self.training_error = {'knn':[],'SVM':[],'DT':[]}
        self.testing_error = {'knn':[],'SVM':[],'DT':[]}

    def record(self,ml_model,set_type,score):
        if set_type == 'train':
            self.training_accuracy[ml_model].append(score)
        elif set_type == 'test':
            self.testing_accuracy[ml_model].append(score)
        elif set_type == 'train error':
            self.training_error[ml_model].append(score)
        else:
            self.testing_error[ml_model].append(score)

    def getRecords(self,ml_model,param):
        if param == 'all':
            return self.stats[ml_model]
        else:
            return self.stats[ml_model][param]

    def getRecord(self,ml_model,param,set_type):
        if set_type == 'train':
            return self.training_accuracy[ml_model][param]
        elif set_type == 'test':
            return self.testing_accuracy[ml_model][param]
        elif set_type == 'train error':
            return self.training_error[ml_model][param]
        else:
            return self.testing_error[ml_model][param]
        
    def evaluate(self,true_labels,predicted_set):
        '''
        This function is to compare the predicted labels with the actual labels and calculate the accuracy score
        '''
        score = round(accuracy_score(true_labels,predicted_set)*100,2)
        error = round(sqrt(mean_squared_error(true_labels,predicted_set)),4)
        return [score,error]
        
    def doStatistics(self,params,train_predicted,train_score,test_predicted,test_score):
        #stats = {} #e.g. stats = { '1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}
        labels = ['malignant','benign']

        train_stats = {n: v for n, v in zip(labels, np.bincount(train_predicted))}
        train_stats['score']=train_score[0]
        train_stats['error']=train_score[1]
        self.train_stats[params[0]].update({str(params[1]):train_stats}) #e.g. train_stats['knn']['1'] = {'benign':x,'malignant':y, 'score':90%}

        test_stats = {n: v for n, v in zip(labels, np.bincount(test_predicted))}
        test_stats['score']=test_score[0]
        test_stats['error']=test_score[1]
        self.test_stats[params[0]].update({str(params[1]):test_stats})

        #e.g. stats['knn']['1'] = {'test': {'m': 132, 'b': 250, 's': 94}, 'train': {'m': 135, 'b': 247, 's': 93}}
        self.stats[params[0]].update({str(params[1]):{'train':train_stats}})
        self.stats[params[0]].get(str(params[1])).update({'test':test_stats})
        #return self.stats
    
    def printReport(self,ml_model,model_factor):
        report_lines = []
        if ml_model == "knn":
            report_lines.append("ML Model: K-Nearest Neighbors")
            report_lines.append("Number of neighbors: "+ str(model_factor))
        elif ml_model == "SVM":
            report_lines.append("ML Model: Support Vector Machine")
            report_lines.append("Kernel Function: "+ str(model_factor))
        else:
            report_lines.append("ML Model: Decision Tree")
            report_lines.append("Max Depth of the Tree: "+ str(model_factor))
        report_lines.append("\t* Training's prediction counts and score: \n\t\t"+str(self.stats[ml_model][str(model_factor)]['train']))
        report_lines.append("\t* Testing's prediction counts and score: \n\t\t"+str(self.stats[ml_model][str(model_factor)]['test']))
        report_lines.append("_"*50)
        for line in report_lines:
            print(line)

    def findBestModel(self,ml_model):
        train_acc = self.training_accuracy[ml_model]
        test_acc = self.testing_accuracy[ml_model]
        ave_train = mean(train_acc)
        ave_test = mean(test_acc)
        print("Average Train Accuracy: {:.2f}%.".format(ave_train))
        print("Average Test Accuracy: {:.2f}%.".format(ave_test))

        best_train = train_acc.index(max(train_acc))
        best_test = test_acc.index(max(test_acc))
        print("Best Training Accuracy is {}% at {}".format(train_acc[best_train],best_train+1))
        print("Best Testing Accuracy is {}% at {}".format(test_acc[best_test],best_test+1))
        
        rmse1 = self.training_error[ml_model]
        rmse2 = self.testing_error[ml_model]
        
        ideal_gap1 = (train_acc[best_train]-train_acc[0])
        ideal_gap2 = (test_acc[best_test]-test_acc[0])
        second_best = best_test

        if best_test == best_train: # ideal case
            return best_test
        else: # better than average case (i.e. no overfitting or underfitting)
            for i in range(1,len(train_acc),1):
                better_train = train_acc[i]-ave_train
                better_test = test_acc[i]-ave_test 
                ave_gap1 = better_train > 0.0 # better than ave
                ave_gap2 = better_test > 0.0 # better than ave
                normal_fitting = abs(train_acc[i]-test_acc[i])<2.0
                
                if (ave_gap1 or ave_gap2) and normal_fitting: # if (better than ave) is true
                    # then either when accuracy is closer to best
                    #second_best = i
                    near_best1 = (train_acc[best_train]-train_acc[i])
                    near_best2 = (test_acc[best_test]-test_acc[i])
                
                    #ideal_rmse = (rmse1[i]+rmse2[i]) < 0.6

                    if (ideal_gap1 > near_best1 or ideal_gap2 > near_best2): 
                        #second best model is at index i
                        ideal_gap1 = near_best1
                        ideal_gap2 = near_best2
                        second_best = i
                
                """ if (ave_gap1 or ave_gap2) and normal_fitting: # if (better than ave) is true
                    # then either when accuracy is closer to best
                    best_model_index = second_best """
            #best_model_index = round((best_test+best_train)/2)
            return second_best
        
    def cleanUp(self,ml_model):
        self.training_accuracy[ml_model] = [] #restart record for the next ML model version
        self.testing_accuracy[ml_model] = [] #restart record for the next ML model version


    
