import numpy as np
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
#from sklearn.metrics import accuracy_score

class NN:
    def __init__(self):
        # Setting up layers of the neural networks
        self.model = keras.Sequential([
                    keras.layers.Flatten(input_shape=(30,)),
                    keras.layers.Dense(20, activation='relu'),
                    keras.layers.Dense(2, activation='sigmoid')
        ])
        # setting up features for compiling
        self.opt = keras.optimizers.Adam(learning_rate=0.01)
        self.loss = 'sparse_categorical_crossentropy'
    
    def changeOpt(self,new_opt):
        self.opt = new_opt

    def changeLoss(self,new_loss):
        self.loss = new_loss

    def compile(self):
        self.model.compile(loss=self.loss,optimizer=self.opt,metrics=['accuracy'])

    def train(self,X_train_std,Y_train,validation_split,epochs):
        self.model.fit(X_train_std,Y_train,validation_split=validation_split,epochs=epochs)
        #return learned
    
    def predict(self,dataset):
        y_pred = self.model.predict(dataset) #this return probability of each class of the dataset
        labels = [np.argmax(i) for i in y_pred]
        targets = []
        for i in labels:
            if i == 0:
                targets.append('malignant')
            else:
                targets.append('benign')
        return targets

    def evaluate(self,X_test_std,Y_test):
        _ , accuracy = self.model.evaluate(X_test_std,Y_test)
        return round(accuracy*100,2)
    
    