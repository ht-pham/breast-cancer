import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from nn import NN
from Data import Data
from FeaturedData import FeaturedData

# Define the only function
def build(grid_settings,X_train,X_test,y_train,y_test):
    grid_search = GridSearchCV(grid_settings[0],grid_settings[1],cv=grid_settings[2],scoring=grid_settings[3])
    grid_search.fit(X_train, y_train)

    ml_algo = type(grid_settings[0]).__name__
    #print(ml_algo)
    # Best Model
    best_model = grid_search.best_estimator_

    match ml_algo:
        case "KNeighborsClassifier":
            # Save the model to a file
            joblib.dump(best_model, 'knn_model.pkl')
            print("Model saved to 'knn_model.pkl'")
        case "SVC":
            joblib.dump(best_model, 'svm_model.pkl')
            print("Model saved to 'svm_model.pkl'")
        case "DecisionTreeClassifier":
            joblib.dump(best_model, 'dt_model.pkl')
            print("Model saved to 'dt_model.pkl'")
    # Predictions
    y_learned = best_model.predict(X_train)
    y_pred = best_model.predict(X_test)
    # Evaluation
    print("Best Parameters:")
    print(grid_search.best_params_)

    print("Accuracy Score on train: {}%".format(round(accuracy_score(y_train, y_learned)*100,4)))
    print("Confusion Matrix Report:")
    print(confusion_matrix(y_test,y_pred))

    print("Accuracy Score on test: {}%".format(round(accuracy_score(y_test, y_pred)*100,4)))
    print("Confusion Matrix Report:")
    print(confusion_matrix(y_test,y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
def buildNeuralNetwork(neural,X_train_std,X_test_std,y_train,y_test):
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
    train_acc = neural.evaluate(X_train_std,y_train)
    test_acc = neural.evaluate(X_test_std,y_test)
    print("Accuracy scores of Train VS Test: {}% <> {}%".format(train_acc,test_acc))

def buildBasicModels(X_train,X_test,y_train,y_test):
    # Set params_grid
    knn_params = {
        'n_neighbors':list(range(1,11,1))
    }
    svm_params = {
        'kernel':['linear','poly','rbf','sigmoid']
    }
    tree_params = {
        'max_depth':[None,10,9,8,7,6,5,4,3,2,1],
        'criterion':["gini", "entropy", "log_loss"]
    }

    print('='*50,' K-Nearest Neighbors Classifier','='*50)
    knn = KNeighborsClassifier()
    grid_settings = [knn, knn_params, 5,'accuracy']
    build(grid_settings,X_train,X_test,y_train,y_test)

    print('='*50,' Support Vector Classifier','='*50)
    svc = SVC(random_state=42)
    grid_settings = [svc, svm_params, 5,'accuracy']
    build(grid_settings,X_train,X_test,y_train,y_test)

    print('='*50,' Decision Tree Classifier','='*50)
    tree = DecisionTreeClassifier(random_state=0)
    grid_settings = [tree, tree_params, 5,'accuracy']
    build(grid_settings,X_train,X_test,y_train,y_test)

# Get data
print("_*_*"*10," Case 1: Original dataset, no change on feature","_*_*"*10)
X_train,X_test,y_train,y_test = Data().split()
## Build knn, svm, and decision tree models with original dataset
buildBasicModels(X_train,X_test,y_train,y_test)
## Build Neural Network based on the original set
### First step: Standardize data for the network
X_train_std,X_test_std = Data().standardize(X_train,X_test)
### Second step: Build the neural network
neural = NN()
buildNeuralNetwork(neural,X_train_std,X_test_std,y_train,y_test)

print("_*_*"*10," Case 2: Dataset with 20 selected features","_*_*"*10)
dataset = FeaturedData()
outliers = ['worst radius','worst texture','worst perimeter','worst area','worst smoothness',
                'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']
dataset.dropFeatures(outliers)
print("*** Dataset has been removed the 'worst' features ","."*50)
X_train,X_test,y_train,y_test = dataset.split(42)
## Build knn, svm, and decision tree models on new dataset
buildBasicModels(X_train,X_test,y_train,y_test)
## Build Neural Network based on the new set
### First step: Standardize data for the network
X_train_std,X_test_std = dataset.standardize(X_train,X_test)
### Second step: Build the neural network
nn2 = NN((20,))
buildNeuralNetwork(nn2,X_train_std,X_test_std,y_train,y_test)

dataset = FeaturedData()
print("_*_*"*10," Case 3: Dataset with 10 selected features","_*_*"*10)
dataset.dropFeatures() 
print("*** Dataset has been removed all errors and worst features ","."*50)
X_train,X_test,y_train,y_test = dataset.split(42)

## Build knn, svm, and decision tree models on new dataset
buildBasicModels(X_train,X_test,y_train,y_test)
## Build Neural Network based on the new dataset
### First step: Standardize data for the network
X_train_std,X_test_std = dataset.standardize(X_train,X_test)
nn1 = NN((10,))
buildNeuralNetwork(nn1,X_train_std,X_test_std,y_train,y_test)