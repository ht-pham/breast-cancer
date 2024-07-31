from KNNModel import KNNModel
import numpy as np

ml_model = KNNModel()
labels = ml_model.dataset.target_names
stats = {} # stats = { '1':{'train':{'benign':x,'malignant':y, 'score':90%},'test':{'score':91%,etc.}}
def doStatistics(n_neighbors,train_predicted,train_score,test_predicted,test_score):
    train_stats = {n: v for n, v in zip(labels, np.bincount(train_predicted))}
    train_stats['score']=train_score
    test_stats = {n: v for n, v in zip(labels, np.bincount(test_predicted))}
    test_stats['score']=test_score
    stats[n_neighbors] = {}
    stats[n_neighbors]['train'] = train_stats
    stats[n_neighbors]['test'] = test_stats
    
def report(stats,model_number):
    report_lines = []
    report_lines.append("Number of neighbors: "+ str(model_number))
    report_lines.append("\t* Training's prediction counts and score: \n\t\t"+str(stats[model_number]['train']))
    report_lines.append("\t* Testing's prediction counts and score: \n\t\t"+str(stats[model_number]['test']))
    report_lines.append("_"*50)
    for line in report_lines:
        print(line)

if __name__ == "__main__":
    ml_model.desc()
    X_train,X_test,y_train,y_test = ml_model.split()
    neighbor_settings = range(1,11,1)
    training_accuracy = []
    testing_accuracy = []
    
    for n in neighbor_settings:
        # Build the model
        ml_model.train(n,X_train,y_train)
        # Predict
        train_pred = ml_model.pred(X_train)
        training_score = ml_model.evaluate(y_train,train_pred)
        training_accuracy.append(training_score)
        
        # Evaluate
        test_pred = ml_model.pred(X_test)
        testing_score = ml_model.evaluate(y_test,test_pred)
        testing_accuracy.append(testing_score)

        doStatistics(n,train_pred,training_score,test_pred,testing_score)
        report(stats,n)