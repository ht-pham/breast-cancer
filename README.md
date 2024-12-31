# Breast Cancer Diagnosis using ML algorithms

### Disclaimer:
This project's main point is to understand different ML algorithms and their accuracy scores **compared with the labels** from the *used* datasets under the various settings on the classifiers. DO NOT intend to use this to diagnose real-life breast cancer cases.

## Dataset
The dataset is from *sk-learn*'s dataset.
* Number of instances/data points: 569
* Number of features: 30
    * Feature names: *mean radius*,*mean texture*, *mean perimeter*, *mean area*, *mean smoothness*,*mean compactness*, *mean concavity*,*mean concave points*, *mean symmetry*, *mean fractal dimension*, *radius error*, *texture error* *perimeter error*, *area error*, *smoothness error*, *compactness error*, *concavity error*, *concave points error*, *symmetry error*, *fractal dimension error*, *worst radius*, *worst texture*, *worst perimeter*, *worst area*,*worst smoothness*, *worst compactness*, *worst concavity*, *worst concave points*, *worst symmetry*, *worst fractal dimension*
* Sample counts per class: 
    * *malignant*: 212
    * *benign*: 357

## Exploratory Data Analysis
As of 08/09, **manual** feature selection has been applied so there are 3 major case study
1. **Case 1**: No features dropped/removed from training
2. **Case 2**: Only major features in use, errors and outliers are removed (i.e. 20 features dropped)
3. **Case 3**: Major features and the error values in use; outliers are removed (i.e. 10 features dropped)

## ML algorithms in use

1. **K-Nearest Neighbors** with number of neighbors from 1 to 10
2. **Suport Vector Machine** with 4 different kernel functions - linear, poly, rbf, and sigmoid
3. **Decision Tree** with unlimited, 10 or less deep 

All of these ML algorithms and their variances will be applied to 3 case studies.

> My hypothesis on the best model is the one whose training's accuracy score and whose testing's accuracy score are ones of the best scores among the other modified model versions and the difference between these scores is the smallest.

> My hypothesis on the second best model is the one whose training's accuracy score and whose testing's accuracy score are higher than average accuracy scores (i.e. train accuracy and test accuracy) compared with other varied models within the same ML algorithm approaches and the difference between the two scores is less than 1%.

## Evaluation
So far, Linear Support Vector Machine is the best ML approach to this categorical problem among 3 ML approaches.
1. **Case 1**: No features dropped/removed from training
> Linear Support Vector Machine is the best ML approach when all features are used

| Best Model                          | Training Accuracy % | Testing Accuracy % |    Case of Fitting   | 
| :-----------------------------------| :-----------------: | :----------------: |:---------------------|
| 10-Nearest Neighbors                |       93.66 %       |       94.41%       |  Right               |
| Linear Support Vector Classifier    |       96.71 %       |       95.8 %       |  Right               |
| 5-Deep Decision Tree with entropy   |       99.30 %       |       94.41 %      |  Light Overfitting   |
| Neural Network with 3 layers        |       98.83 %       |       95.8 %       |  Light Overfitting   |

2. **Case 2**: Only major features in use, errors and outliers are removed (i.e. 20 features dropped)
   
> 4-Deep Decision Tree with entropy criterion is is the best ML approach when only 10 features in used

| Best Model                          | Training Accuracy % | Testing Accuracy % |    Case of Fitting   |
| :-----------------------------------| :-----------------: | :----------------: |:---------------------|
| 3-Nearest Neighbors                 |       91.78 %       |       90.21 %      |  Right               |
| Linear Support Vector Classifier    |       91.31 %       |       94.41 %      |  Light Underfitting  |
| 4-Deep Decision Tree with entropy   |       95.07 %       |       95.11 %      |  Right               |
| Neural Network with 3 layers        |       95.54 %       |       97.9  %      |  Light Undefitting   |

3. **Case 3**: Major features and the error values in use; outliers are removed (i.e. 10 features dropped)
> Linear Support Vector Machine is the best ML approach when 20 features in use.

| Best Model                          | Training Accuracy % | Testing Accuracy % |   Case of Fitting   |
| :-----------------------------------| :-----------------: | :----------------: |:--------------------|
| 9-Nearest Neighbors                 |       88.96 %       |       91.6 %       | Right               |
| Linear Support Vector Classifier    |       93.66 %       |       94.41 %      | Right               |
| 1-Deep Decision Tree with gini      |       92.25 %       |       89.51 %      | Right               |
| Neural Network with 3 layers        |       98.36 %       |       95.1  %      | Light Overfitting   |

