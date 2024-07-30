import lib
import numpy as np

#cancer = lib.dataset

if __name__ == "__main__":
    lib.desc()
    X_train,X_test,y_train,y_test = lib.split()
    model = lib.train(X_train,y_train)
    train_pred = lib.pred(X_train,model)
    print("Counts per class of predicted results:\n",{n: v for n, v in zip(lib.dataset.target_names, np.bincount(train_pred))})
    print("Training's Accuracy score: ", round(lib.evaluate(X_train,y_train,model)*100,2),"%")
    test_pred = lib.pred(X_test,model)
    print("Counts per class of predicted results:\n",{n: v for n, v in zip(lib.dataset.target_names, np.bincount(test_pred))})
    print("Testing's Accuracy score: ",round(lib.evaluate(X_test,y_test,model)*100,2),"%")