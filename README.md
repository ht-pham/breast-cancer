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
3. **Decision Tree** with unlimited, 10 or less deep (Best: 1-deep)

> My hypothesis of the best model is the one whose training's accuracy score and whose testing's accuracy score are the best scores among the other modified model versions and the difference between these scores is the smallest.

So far, linear SVM is the best ML approach to this categorical problem among 3 ML approaches.
| Best Model                          | Training Accuracy % | Testing Accuracy % |
| :-----------------------------------| :-----------------: | :----------------: |
| 8-Nearest Neighbors                 |       94.13 %       |       94.41%       |
| Linear Support Vector Machine       |       96.71 %       |       95.8 %       |
| Decision Tree                       |       92.25 %       |       92.31 %      |
