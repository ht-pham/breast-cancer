import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from Data import Data

class FeaturedData(Data):
    def __init__(self):
        super().__init__()

    def desc(self):
        super().desc()

    def getLabels(self):
        return super().getLabels()
    
    def split(self,state):
        return super().split(state)
    
    def dropFeatures(self,cols=['radius error','texture error','perimeter error','area error','smoothness error',
                      'compactness error','concavity error','concave points error','symmetry error',
                      'fractal dimension error','worst radius','worst texture','worst perimeter','worst area',
                      'worst smoothness','worst compactness','worst concavity','worst concave points',
                      'worst symmetry','worst fractal dimension']):
        self.df = self.df.drop(cols,axis="columns")
        
        #return self.df
        
    
    

    
