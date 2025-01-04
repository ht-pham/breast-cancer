from flask import Flask, request, redirect, url_for, render_template
import joblib
import numpy as np


# Load the trained ML model
knn = joblib.load('knn_model.pkl')  
svm = joblib.load('svm_model.pkl')  
tree = joblib.load('dt_model.pkl')  
neural = joblib.load('nn_model.pkl')


# Initialize Flask app
app = Flask(__name__)

# define global variable
features = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity',
                'Concave points','Symmetry','Fractal dimension']
@app.route("/")
def welcome():
    return render_template("homepage.html",features=features)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    user_input = {feature:request.form.get(feature,type=float) for feature in features}
    if not user_input:
        return redirect(url_for('welcome'))

    # Key mapping to features 
    df_features = ['mean radius','mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension']
    key_mapping = dict(zip(features,df_features))
    renamed_input = {key_mapping.get(k):v for k,v in user_input.items()}
    
    # Convert query parameter values to float and reshape
    reshaped_input = np.array([float(value) for value in renamed_input.values()]).reshape(1,-1)

    # Make prediction
    predicts=[]
    predicts.append(knn.predict(reshaped_input)[0])
    predicts.append(svm.predict(reshaped_input)[0])
    predicts.append(tree.predict(reshaped_input)[0])
    predicts.append(neural.predict(reshaped_input)[0][-1])
    
    res=[castVote(predictions=predicts)]
    res.extend(list(map(isBenign,predicts)))

    """ results = {
        'knn_prediction': knn_prediction,
        'knn_confidence': 85.2,
        'knn_accuracy': 90.5,
        'svm_prediction': svm_prediction,
        'svm_confidence': 88.7,
        'svm_accuracy': 92.0,
        'decision_tree_prediction': dt_prediction,
        'decision_tree_feature_importance': 'Radius Mean: 0.45, Texture Mean: 0.30',
        'decision_tree_accuracy': 89.8,
        'neural_network_prediction': nn_prediction,
        'neural_network_confidence': 92.4,
        'neural_network_accuracy': 94.5
    } """

    html_names = ['Overall_prediction','knn_prediction','svm_prediction','decision_tree_prediction','neural_network_prediction']
    results = dict(zip(html_names,res))
    return render_template('result.html', **results)
    
def isBenign(predicted):
    if predicted == 0:
        return "Benign"
    else:
        return "Malignant"
def castVote(predictions):
    # use majority vote rather average threshold to conclude because y_pred returns 0 or 1 
    benign = predictions.count(0)
    malignant = predictions.count(1)
    return "Benign" if benign > malignant else "Malignant"


# Run the app
if __name__ == '__main__':
    app.run(debug=True)