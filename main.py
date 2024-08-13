from report import Report

from Data import Data
from FeaturedData import FeaturedData

from KNNModel import KNNModel
from SVMModel import SVMModel
from DecisionTree import Tree
from nn import NN

# Object creations
# create Report
rp = Report()
# import dataset
dataset = Data()
# Create models
knn = KNNModel()
svm = SVMModel()
tree = Tree()

#create NN


def runKNN(neighbor_settings,X_train,X_test,y_train,y_test):
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

def runSVM(kernels,X_train,X_test,y_train,y_test):
    for kernel in kernels:
        # Stage 1: Train/Fit data into model
        svm.train(kernel,X_train,y_train)
        # Stage 2: Predict & Test
        train_pred = svm.pred(X_train,'train')
        test_pred = svm.pred(X_test,'test')
        # Evaluate
        train_accuracy, test_accuracy = rp.evaluate('SVM',[y_train,train_pred],[y_test,test_pred])
        #Report
        rp.doStatistics(['SVM',kernel],train_pred,train_accuracy,test_pred,test_accuracy)
        rp.printReport("SVM",kernel)
    
    ind = rp.findBestModel("SVM")
    kernel = kernels[ind]
    print("Best SVM model is SVM with kernel function as",kernel
          ,"with ",rp.getRecord("SVM",ind,'train'),"% accuracy in training",
            " and with ",rp.getRecord("SVM",ind,'test'),"% accuracy in testing")
    print('*'*100)
    rp.cleanUp('SVM')
    
def runDT(depths,X_train,X_test,y_train,y_test):
    for i in depths:
        #Train
        tree.train(i,X_train,y_train)
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
        depth = depths[id]
    print("Best Decision Tree is the {}-deep Decision Tree with {:.2f}% accuracy in training, and {:.2f}% accuracy in testing."
          .format(depth,rp.getRecord('DT',id,'train'),rp.getRecord('DT',id,'test')))
    
    print('*'*100)
    rp.cleanUp('DT')

def runNN(X_train,X_test,y_train,y_test):
    #First step: Standardize data
    neural = NN()
    X_train_std,X_test_std = dataset.standardize(X_train,X_test)
    #Second step: set up layers through keras (already did through NN() obj creation)
    #Third step: compile the neural network
    neural.compile()
    #Fourth step: train
    neural.train(X_train_std,y_train,0.1,10)
    #Fifth step: Predict
    y_learned = neural.predict(X_train_std)
    y_pred = neural.predict(X_test_std)
    
    # Evaluate
    malignant = sum(1 for x in y_learned if x == 'malignant')
    benign = sum(1 for x in y_learned if x == 'benign')
    print('Predictions on train set: (1) malignant: {}, (2) benign: {}'.format(malignant,benign))
    malignant = sum(1 for x in y_pred if x == 'malignant')
    benign = sum(1 for x in y_pred if x == 'benign')
    print('Predictions on test set: (1) malignant: {}, (2) benign: {}'.format(malignant,benign))
    #train_accuracy, test_accuracy = rp.evaluate('NN',[y_train,y_learned],[y_test,y_pred])
    train_acc = neural.evaluate(X_train_std,y_train)
    test_acc = neural.evaluate(X_test_std,y_test)
    print("Accuracy scores of Train VS Test: {}% <> {}%".format(train_acc,test_acc))

if __name__ == "__main__":
    dataset.desc()
    # Spliting the dataset
    X_train,X_test,y_train,y_test = dataset.split()

    ### This is kNN model
    print('*','_'*30,"k-nearest neighbors",'_'*30,'*')
    runKNN(range(1,11,1),X_train,X_test,y_train,y_test)
    ### This is SVM model
    print('*','_'*30,"Support Vector Machine",'_'*30,'*')
    runSVM(['linear','poly','rbf','sigmoid'],X_train,X_test,y_train,y_test)
    
    ### Decision Tree Classifier 
    print('*','_'*30,"Decision Tree",'_'*30,'*')
    X_train,X_test,y_train,y_test = dataset.split(random=42,Stratify=dataset.df_target) #Stratify -- special keyword for Tree)
    depth_settings = [None]
    depth_settings[1:]=[i for i in range(10,0,-1)]
    runDT(depth_settings,X_train,X_test,y_train,y_test)

    ### Neural Network
    dataset = Data()
    X_train,X_test,y_train,y_test = dataset.split()
    print('*','_'*30,"Neural Network",'_'*30,'*')
    runNN(X_train,X_test,y_train,y_test)
    
    #------------------------------------------------------------------------------------#
    #------------- ML Algorithms with dataset applied feature selection -----------------# 
    dataset = FeaturedData()
    #---- Case 1: Drop 20 features
    dataset.dropFeatures()
    X_train,X_test,y_train,y_test = dataset.split(42)

    #------------ Using kNN models
    print('*','_'*30,"k-nearest neighbors",'_'*30,'*')
    runKNN(range(1,11,1),X_train,X_test,y_train,y_test)
    #------------ Using SVM models
    print('*','_'*30,"Support Vector Machine",'_'*30,'*')
    runSVM(['linear','poly','rbf','sigmoid'],X_train,X_test,y_train,y_test)
    #------------ Using DT models
    print('*','_'*30,"Decision Tree",'_'*30,'*')
    X_train,X_test,y_train,y_test = dataset.split(random_state=42,Stratify=dataset.df_target)
    runDT(depth_settings,X_train,X_test,y_train,y_test)
    #------------ Using Neural Network
    

    #---- Case 2: Drop outliers only
    dataset = FeaturedData()
    outliers = ['worst radius','worst texture','worst perimeter','worst area','worst smoothness',
                'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']
    dataset.dropFeatures(outliers)
    X_train,X_test,y_train,y_test = dataset.split(42)
    #------------ Using kNN models
    print('*','_'*30,"k-nearest neighbors",'_'*30,'*')
    runKNN(range(1,11,1),X_train,X_test,y_train,y_test)
    #------------ Using SVM models
    print('*','_'*30,"Support Vector Machine",'_'*30,'*')
    runSVM(['linear','poly','rbf','sigmoid'],X_train,X_test,y_train,y_test)
    #------------ Using DT models
    print('*','_'*30,"Decision Tree",'_'*30,'*')
    X_train,X_test,y_train,y_test = dataset.split(random_state=42,Stratify=dataset.df_target)
    runDT(depth_settings,X_train,X_test,y_train,y_test)

    #--------------- Neural Networks ------------------#
    




