from math import sqrt
from statistics import mean
from time import time
import numpy as np
from Data import Data
from report import Report
rp = Report()

from KNNModel import KNNModel
from SVMModel import SVMModel
from DecisionTree import Tree
from FeaturedData import FeaturedData


def findBestModel(train_acc,test_acc):
    best_train = train_acc.index(max(train_acc))
    best_test = test_acc.index(max(test_acc))
    if best_test == best_train: # ideal case
        return best_test
    else: # average case (i.e. no overfitting or underfitting)
        ave_train = mean(train_acc)
        ave_test = mean(test_acc)
        print("Average Train Accuracy: {:.2f}%.".format(ave_train))
        print("Average Test Accuracy: {:.2f}%.".format(ave_test))
        for i in range(0,len(train_acc),1):
            train_gap = train_acc[i]-ave_train
            test_gap = test_acc[i]-ave_test
            ideal_gap1 = train_gap > 1.0 and train_gap < 3.0 # better than ave
            ideal_gap2 = test_gap > 1.0 and test_gap < 3.0 # better than ave
            if ideal_gap1 or ideal_gap2 and abs(train_acc[i]-test_acc[i])<1.0:
                best_model_index = i
        return best_model_index
    

if __name__ == "__main__":
    # Print data description
    dataset = Data()
    dataset.desc()
    # Spliting the dataset
    X_train,X_test,y_train,y_test = dataset.split()

    # Create models
    knn = KNNModel()
    svm = SVMModel()
    tree = Tree()

    ### This is kNN model
    print('*','_'*30,"k-nearest neighbors",'_'*30,'*')
    neighbor_settings = range(1,11,1)
    for n in neighbor_settings:
        # Build the model
        knn.train(n,X_train,y_train)
        # Predict
        train_pred = knn.pred(X_train,'train')
        test_pred = knn.pred(X_test,'test')
        # Evaluate
        train_accuracy, test_accuracy = rp.evaluate('knn',[y_train,train_pred],[y_test,test_pred])
        rp.doStatistics(['knn',n],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport("knn",n)
        
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    i = rp.findBestModel("knn")
    print("Best kNN Model is",str(i+1)+"-nearest neighbors with"
          ,rp.getRecord("knn",i,'train'),"% of accuracy in training and"
          ,rp.getRecord("knn",i,'test'),"% of accuracy in testing")
    print('*'*100)
    rp.cleanUp('knn')
        
    ### This is SVM model
    #svm.desc()
    print('*','_'*30,"Support Vector Machine",'_'*30,'*')
    #X_train,X_test,y_train,y_test = svm.split()
    kernel_settings = ['linear','poly','rbf','sigmoid']
    
    for kernel in kernel_settings:
        # Stage 1: Train/Fit data into model
        print('* Start training ')
        start = time()
        svm.train(kernel,X_train,y_train)
        end = time()
        print('* Finished training')
        print("Total elapsed time for training: {:.8f}\n".format(end-start))
        
        # Stage 2: (still training) Make prediction against training
        print('* Predicting on training data...')
        start = time()
        train_pred = svm.pred(X_train)
        end = time()
        print('* Finished predicting ')
        print("Total elapsed time for predicting all datapoints: {:.8f}\n".format(end-start))
        ## Evaluate the predicted results against training set
        train_accuracy = rp.calculateMetrics(y_train,train_pred)
        rp.record('SVM','train',train_accuracy[0])
        rp.record('SVM','train error',train_accuracy[1])
        #svm.calculateMetrics('train',y_train,train_pred)
        #rp.record("SVM",'train',svm.getScore('train'))
        
        # Stage 3: Evaluate against testing set
        print('* Predicting against testing set...')
        start = time()
        test_pred = svm.pred(X_test)
        end = time()
        print('* Finished predicting')
        print("Total elapsed time for testing: {:.8f}\n".format(end-start))
        ## Evaluate the predicted results against testing set
        test_accuracy = rp.calculateMetrics(y_test,test_pred)
        rp.record('SVM','test',test_accuracy[0])
        rp.record('SVM','test error',test_accuracy[1])
        #svm.calculateMetrics('test',y_test,test_pred)
        #rp.record("SVM",'test',svm.getScore('test'))
        #Report
        rp.doStatistics(['SVM',kernel],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport("SVM",kernel)
    
    ind = rp.findBestModel("SVM")
    kernel = kernel_settings[ind]
    print("Best SVM model is SVM with kernel function as",kernel
          ,"with ",rp.getRecord("SVM",ind,'train'),"% accuracy in training",
            " and with ",rp.getRecord("SVM",ind,'test'),"% accuracy in testing")
    
    ### Decision Tree Classifier
    print('*','_'*30,"Decision Tree",'_'*30,'*')
    X_train,X_test,y_train,y_test = dataset.split(random=42,Stratify=dataset.df_target)
    depth_settings = [None]
    depth_settings[1:]=[i for i in range(10,0,-1)]
    
    for i in depth_settings:
        #Train
        start = time()
        tree.train(i,X_train,y_train)
        if i == None:
            i = 'Infinite'
        print("* Finished training model with {}-depth within {:.8f}".format(i,time()-start))
        
        # Predict
        train_pred = tree.pred(X_train,'train')
        test_pred = tree.pred(X_test,'test')
        # Evaluate
        train_accuracy, test_accuracy = rp.evaluate('DT',[y_train,train_pred],[y_test,test_pred])

        rp.doStatistics(['DT',i],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport('DT',i)

    id = rp.findBestModel('DT')
    if id==0:
        depth="Infinite"
    else:
        depth = depth_settings[id]
    print("Best Decision Tree is the {}-deep Decision Tree with {:.2f}% accuracy in training, and {:.2f}% accuracy in testing."
          .format(depth,rp.getRecord('DT',id,'train'),rp.getRecord('DT',id,'test')))
    
    print('*'*100)
    #------------------------------------------------------------------------------------#
    #------------- ML Algorithms with dataset applied feature selection -----------------# 
    dataset = FeaturedData()
    #---- Case 1: Drop 20 features
    dataset.dropFeatures()
    X_train,X_test,y_train,y_test = dataset.split(42)
    
    for n in neighbor_settings:
        # Build the model
        knn.train(n,X_train,y_train)
        # Predict
        train_pred = knn.pred(X_train,'train')
        test_pred = knn.pred(X_test,'test')
        # Evaluate
        train_accuracy, test_accuracy = rp.evaluate('knn',[y_train,train_pred],[y_test,test_pred])
        rp.doStatistics(['knn',n],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport("knn",n)

    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    id = rp.findBestModel("knn")
    print("Best kNN Model for dataset with only 10 selected features is",str(id+1)+"-nearest neighbors with"
          ,rp.getRecord("knn",id,'train'),"% of accuracy in training and"
          ,rp.getRecord("knn",id,'test'),"% of accuracy in testing")
    print('*'*100)
    rp.cleanUp('knn')

    #---- Case 2: Drop outliers only
    dataset = FeaturedData()
    outliers = ['worst radius','worst texture','worst perimeter','worst area','worst smoothness',
                'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']
    dataset.dropFeatures(outliers)
    X_train,X_test,y_train,y_test = dataset.split(42)
    
    for n in neighbor_settings:
        # Build the model
        knn.train(n,X_train,y_train)
        # Predict
        train_pred = knn.pred(X_train,'train')
        test_pred = knn.pred(X_test,'test')
        # Evaluate
        train_accuracy, test_accuracy = rp.evaluate('knn',[y_train,train_pred],[y_test,test_pred])
        rp.doStatistics(['knn',n],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport("knn",n)
        
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    i = rp.findBestModel("knn")
    print("Best kNN Model for dataset with 20 selected features is",str(i+1)+"-nearest neighbors with"
          ,rp.getRecord("knn",i,'train'),"% of accuracy in training and"
          ,rp.getRecord("knn",i,'test'),"% of accuracy in testing")
    print('*'*100)
    rp.cleanUp('knn')
   






