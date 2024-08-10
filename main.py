from math import sqrt
from statistics import mean
from time import time
import numpy as np
from report import Report
rp = Report()

from KNNModel import KNNModel
from SVMModel import SVMModel
from DecisionTree import Tree
from FeaturedData import FeaturedData
knn = KNNModel()
svm = SVMModel()
tree = Tree()


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
    
    ### This is kNN model
    print('*'*100)
    knn.desc()
    print('*','_'*30,"k-nearest neighbors",'_'*30,'*')
    X_train,X_test,y_train,y_test = knn.split() 
    neighbor_settings = range(1,11,1)
    training_accuracy = []
    testing_accuracy = []
    
    for n in neighbor_settings:
        # Build the model
        print("* Start building & training")
        start = time()
        knn.train(n,X_train,y_train)
        end = time()
        train_time = end - start
        print("* Done building & training")
        # Predict
        print("* Start predicting against training set")
        start = time()
        train_pred = knn.pred(X_train)
        end = time()
        pred_time = end - start
        print("* Done with training stage")
        training_score = knn.evaluate(y_train,train_pred)
        training_accuracy.append(training_score)
        
        # Evaluate
        print("* Start evaluating")
        start = time()
        test_pred = knn.pred(X_test)
        end = time()
        test_time = end - start
        print("* Done with evaluation")
        testing_score = knn.evaluate(y_test,test_pred)
        testing_accuracy.append(testing_score)

        rp.doStatistics(['knn',n],train_pred,training_score,test_pred,testing_score)
        print("* Total elapsed time for building & training the model: {:.8f}".format(train_time))
        print("* Total elapsed time for testing against the training model: {:.8f}".format(pred_time))
        print("* Total elapsed time for evaluating the model: {:.8f}\n".format(test_time))

        rp.printReport("knn",n)
        
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    best_model_index = findBestModel(training_accuracy,testing_accuracy)
    print("Best kNN Model is",str(best_model_index+1)+"-nearest neighbors with"
          ,training_accuracy[best_model_index],"% of accuracy in training and"
          ,testing_accuracy[best_model_index],"% of accuracy in testing")
    print('*'*100)
        
    ### This is SVM model
    #svm.desc()
    print('*','_'*30,"Support Vector Machine",'_'*30,'*')
    X_train,X_test,y_train,y_test = svm.split()
    kernel_settings = ['linear','poly','rbf','sigmoid']
    training_accuracy = {}
    testing_accuracy = {}
    # reset training score and testing score lists
    training_score = []
    testing_score = []
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
        svm.evaluate('train',y_train,train_pred)
        training_accuracy[kernel]= {
            'confusion matrix':svm.evaluation['train']['Confusion Matrix'],
            'score':svm.evaluation['train']['score']}
        training_score.append(svm.evaluation['train']['score'])
        # Stage 3: Evaluate against testing set
        print('* Predicting against testing set...')
        start = time()
        test_pred = svm.pred(X_test)
        print('* Finished predicting')
        end = time()
        print("Total elapsed time for testing: {:.8f}\n".format(end-start))
        ## Evaluate the predicted results against testing set
        testing_eval = svm.evaluate('test',y_test,test_pred)
        testing_accuracy[kernel]= {
            'confusion matrix':svm.evaluation['test']['Confusion Matrix'],
            'score':svm.evaluation['test']['score']}
        testing_score.append(svm.evaluation['test']['score'])
        #Report
        rp.doStatistics(['SVM',kernel],train_pred,svm.evaluation['train']['score'],test_pred,svm.evaluation['test']['score'])
        rp.printReport("SVM",kernel)
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    best_model = findBestModel(training_score,testing_score)
    kernel = kernel_settings[best_model]
    print("Best SVM model is SVM with kernel function as",kernel
          ,"with ",training_score[best_model],"% accuracy in training",
            " and with ",testing_score[best_model],"% accuracy in testing")
    
    ### Decision Tree Classifier
    print('*','_'*30,"Decision Tree",'_'*30,'*')
    X_train,X_test,y_train,y_test = tree.split()
    
    accuracy_stats = {}
    print("Doing overfitting case i.e. infinite depth")
    print('* Start training ')
    start = time()
    tree.train(None,X_train,y_train)
    end = time()
    print("Finishing training after {:.8f}".format(end-start))
    print("* Predicting with sample sets")
    start = time()
    train_pred = tree.pred(X_train)
    end = time()
    print("... training set: done within {:.8f}".format(end-start))
    start = time()
    test_pred = tree.pred(X_test)
    end = time()
    print("... testing set: done within {:.8f}".format(end-start))

    accuracy_stats.update({'Infinite':{'train':tree.evaluate('train acc',y_train,train_pred),
                              'test':tree.evaluate('test acc',y_test,test_pred)}})
    
    depth_settings = range(10,0,-1) # Descending because of max depth from infinite to 1
    for i in depth_settings:
        #Train
        start = time()
        tree.train(i,X_train,y_train)
        print("* Finished training model with {}-depth within {:.8f}s".format(i,time()-start))
        print("Doing some predictions")

        #Predict
        print("* Predicting with sample sets")
        start = time()
        train_pred = tree.pred(X_train)
        end = time()
        print("... training set: done within {:.8f}".format(end-start))
        #Test
        start = time()
        test_pred = tree.pred(X_test)
        end = time()
        print("... testing set: done within {:.8f}".format(end-start))

        #Evaluate
        train_score = tree.evaluate('train acc',y_train,train_pred)
        test_score = tree.evaluate('test acc',y_test,test_pred)
        rp.doStatistics(['DT',i],train_pred,train_score,test_pred,test_score)
        rp.printReport("DT",i)

        accuracy_stats.update({str(i):{'train':train_score,'test':test_score}})
    
    training_accuracy = [accuracy_stats['Infinite']['train'],]
    testing_accuracy = [accuracy_stats['Infinite']['test'],]
    for k,v in accuracy_stats.items():
        if k == 'Infinite':
            continue
        for l,s in v.items():
            if l == 'train':
                training_accuracy.append(s)
            else:
                testing_accuracy.append(s)

    best_model_index = findBestModel(training_accuracy,testing_accuracy)
    if best_model_index == 0:
        print("Best Decision Tree is a deep rooted Decision Tree"+
              "with unlimited depth,"+
              "{:.2f}% accuracy in training, and{:.2f}% accuracy in testing".format(training_accuracy[0],testing_accuracy[0]))
    else:
        depth = 11-best_model_index
        print("Best Decision Tree is Decision Tree with max depth of {},".format(depth))
        print("{:.2f}% accurate in training, and {:.2f}% accurate in testing".format(accuracy_stats[str(depth)]['train'],accuracy_stats[str(depth)]['test']))


    #------------------------------------------------------------------------------------#
    #------------- ML Algorithms with dataset applied feature selection -----------------# 
    dataset = FeaturedData()
    #---- Case 1: Drop 20 features
    dataset.dropFeatures()
    X_train,X_test,y_train,y_test = dataset.split()
    # Model 1: k-Nearest Neighbors
    neighbor_settings = range(1,11,1)
    training_accuracy = []
    testing_accuracy = []
    
    for n in neighbor_settings:
        # Build the model
        print("* Start building & training")
        start = time()
        knn.train(n,X_train,y_train)
        end = time()
        train_time = end - start
        print("* Done building & training")
        # Predict
        print("* Start predicting against training set")
        start = time()
        train_pred = knn.pred(X_train)
        end = time()
        pred_time = end - start
        print("* Done with training stage")
        training_score = knn.evaluate(y_train,train_pred)
        training_accuracy.append(training_score)
        
        # Evaluate
        print("* Start evaluating")
        start = time()
        test_pred = knn.pred(X_test)
        end = time()
        test_time = end - start
        print("* Done with evaluation")
        testing_score = knn.evaluate(y_test,test_pred)
        testing_accuracy.append(testing_score)

        rp.doStatistics(['knn',n],train_pred,training_score,test_pred,testing_score)
        print("* Total elapsed time for building & training the model: {:.8f}".format(train_time))
        print("* Total elapsed time for testing against the training model: {:.8f}".format(pred_time))
        print("* Total elapsed time for evaluating the model: {:.8f}\n".format(test_time))

        rp.printReport("knn",n)
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    best_model_index = findBestModel(training_accuracy,testing_accuracy)
    print("Best kNN Model is",str(best_model_index+1)+"-nearest neighbors with"
          ,training_accuracy[best_model_index],"% of accuracy in training and"
          ,testing_accuracy[best_model_index],"% of accuracy in testing")
    print('*'*100)

    #---- Case 1: Drop outliers only
    dataset = FeaturedData()
    outliers = ['worst radius','worst texture','worst perimeter','worst area','worst smoothness',
                'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']
    dataset.dropFeatures(outliers)
    X_train,X_test,y_train,y_test = dataset.split()
    # Model 1: k-Nearest Neighbors
    neighbor_settings = range(1,11,1)
    training_accuracy = []
    testing_accuracy = []
    
    for n in neighbor_settings:
        # Build the model
        print("* Start building & training")
        start = time()
        knn.train(n,X_train,y_train)
        end = time()
        train_time = end - start
        print("* Done building & training")
        # Predict
        print("* Start predicting against training set")
        start = time()
        train_pred = knn.pred(X_train)
        end = time()
        pred_time = end - start
        print("* Done with training stage")
        training_score = knn.evaluate(y_train,train_pred)
        training_accuracy.append(training_score)
        
        # Evaluate
        print("* Start evaluating")
        start = time()
        test_pred = knn.pred(X_test)
        end = time()
        test_time = end - start
        print("* Done with evaluation")
        testing_score = knn.evaluate(y_test,test_pred)
        testing_accuracy.append(testing_score)

        rp.doStatistics(['knn',n],train_pred,training_score,test_pred,testing_score)
        print("* Total elapsed time for building & training the model: {:.8f}".format(train_time))
        print("* Total elapsed time for testing against the training model: {:.8f}".format(pred_time))
        print("* Total elapsed time for evaluating the model: {:.8f}\n".format(test_time))

        rp.printReport("knn",n)
        
    # Best model is defined as the one either with almost similiar training score and testing score 
    # or simply best accuracies in both set
    best_model_index = findBestModel(training_accuracy,testing_accuracy)
    print("Best kNN Model is",str(best_model_index+1)+"-nearest neighbors with"
          ,training_accuracy[best_model_index],"% of accuracy in training and"
          ,testing_accuracy[best_model_index],"% of accuracy in testing")
    print('*'*100)
   






