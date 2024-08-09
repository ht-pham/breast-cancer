from time import time
import numpy as np

from KNNModel import KNNModel
from SVMModel import SVMModel

knn = KNNModel()
svm = SVMModel()
labels = knn.dataset.target_names

def doStatistics(params,train_predicted,train_score,test_predicted,test_score):
    stats = {} #e.g. stats = { '1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}
    train_stats = {n: v for n, v in zip(labels, np.bincount(train_predicted))}
    train_stats['score']=train_score
    test_stats = {n: v for n, v in zip(labels, np.bincount(test_predicted))}
    test_stats['score']=test_score
    stats[params] = {}
    stats[params]['train'] = train_stats
    stats[params]['test'] = test_stats
    return stats
    
def report(stats,ml_model,model_factor):
    report_lines = []
    report_lines.append("ML Model: "+ml_model)
    if ml_model == "K-Nearest Neighbor":
        report_lines.append("Number of neighbors: "+ str(model_factor))
    else:
        report_lines.append("Kernel Function: "+ str(model_factor))
    report_lines.append("\t* Training's prediction counts and score: \n\t\t"+str(stats[model_factor]['train']))
    report_lines.append("\t* Testing's prediction counts and score: \n\t\t"+str(stats[model_factor]['test']))
    report_lines.append("_"*50)
    for line in report_lines:
        print(line)

def findBestModel(train_acc,test_acc):
    best_train = train_acc.index(max(train_acc))
    best_test = test_acc.index(max(test_acc))
    if best_test == best_train:
        return best_test
    else:
        min_gap = abs(train_acc[0]-test_acc[0])
        best_model_index = 0
        for i in range(1,len(train_acc),1):
            gap = abs(train_acc[i]-test_acc[i])
            if min_gap > gap:
                min_gap = gap
                best_model_index = i
        return best_model_index
    

if __name__ == "__main__":
    
    ### This is kNN model
    print('*'*100)
    knn.desc()
    print('*','_'*30,"k-nearest-neighbor",'_'*30,'*')
    X_train,X_test,y_train,y_test = knn.split()
    neighbor_settings = range(1,11,1)
    training_accuracy = []
    testing_accuracy = []
    comp_stats = []
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

        stats = doStatistics(n,train_pred,training_score,test_pred,testing_score)
        print("* Total elapsed time for building & training the model: {:.8f}".format(train_time))
        print("* Total elapsed time for testing against the training model: {:.8f}".format(pred_time))
        print("* Total elapsed time for evaluating the model: {:.8f}\n".format(test_time))

        report(stats,"K-Nearest Neighbor",n)
        
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
        stats = doStatistics(kernel,train_pred,svm.evaluation['train']['score'],test_pred,svm.evaluation['test']['score'])
        report(stats,"Support Vector Machine",kernel)
    
    best_model = findBestModel(training_score,testing_score)
    kernel = kernel_settings[best_model]
    print("Best SVM model is SVM with kernel function as",kernel
          ,"with ",training_score[best_model],"% accuracy in training",
            " and with ",testing_score[best_model],"% accuracy in testing")



