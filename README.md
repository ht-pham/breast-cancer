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
As of now, no method is in used

## ML algorithms in use
    1. **K-Nearest Neighbors** with number of neighbors from 1 to 10 (Best: 8-nn)
    2. **Suport Vector Machine** with 4 different kernel functions - linear, poly, rbf, and sigmoid (Best: linear)

    Between kNN and SVM, linear SVM is a better ML approach to this problem.