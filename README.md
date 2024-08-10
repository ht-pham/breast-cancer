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
As of 08/09, feature selection has been applied so there are 3 major case study
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
1. **Case 1**: No features dropped/removed from training
So far, Decision Tree with Depth of 5 is the best ML approach to this categorical problem among 3 ML approaches.
| Best Model                          | Training Accuracy % | Testing Accuracy % | 
| :-----------------------------------| :-----------------: | :----------------: |
| 8-Nearest Neighbors                 |       94.13 %       |       94.41%       |
| Linear Support Vector Machine       |       96.71 %       |       95.8 %       |
| Decision Tree with Depth of 5       |       99.53 %       |       95.10 %      |

2. **Case 2**: Only major features in use, errors and outliers are removed (i.e. 20 features dropped)
> as of now, only knn is used. Average Train Accuracy: 91.36%. Average Test Accuracy: 89.09%.
| Best Model                          | Training Accuracy % | Testing Accuracy % |
| :-----------------------------------| :-----------------: | :----------------: |
| 7-Nearest Neighbors                 |       89.67 %       |       90.21%       |


3. **Case 3**: Major features and the error values in use; outliers are removed (i.e. 10 features dropped)
> as of now, only knn is used. Average Train Accuracy: 91.39%. Average Test Accuracy: 88.81%.
| Best Model                          | Training Accuracy % | Testing Accuracy % |
| :-----------------------------------| :-----------------: | :----------------: |
| 3-Nearest Neighbors                 |       92.96 %       |       90.21%       |
