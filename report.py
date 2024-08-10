import csv
import numpy as np

class Report:
    def __init__(self):
        #e.g. stats = { ml_algo:{'1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}}
        self.stats = {'knn':{},'SVM':{},'DT':{}}
        self.train_stats = {'knn':{},'SVM':{},'DT':{}}
        self.test_stats = {'knn':{},'SVM':{},'DT':{}}

    def doStatistics(self,params,train_predicted,train_score,test_predicted,test_score):
        #stats = {} #e.g. stats = { '1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}
        labels = ['malignant','benign']

        train_stats = {n: v for n, v in zip(labels, np.bincount(train_predicted))}
        train_stats['score']=train_score
        self.train_stats[params[0]].update({str(params[1]):train_stats}) #e.g. train_stats['knn']['1'] = {'benign':x,'malignant':y, 'score':90%}

        test_stats = {n: v for n, v in zip(labels, np.bincount(test_predicted))}
        test_stats['score']=test_score
        self.test_stats[params[0]].update({str(params[1]):test_stats})

        #e.g. stats['knn']['1'] = {'test': {'m': 132, 'b': 250, 's': 94}, 'train': {'m': 135, 'b': 247, 's': 93}}
        self.stats[params[0]].update({str(params[1]):{'train':train_stats}})
        self.stats[params[0]][str(params[1])].update({'test':test_stats})
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

