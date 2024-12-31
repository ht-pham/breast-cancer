from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained ML model
knn = joblib.load('knn_model.pkl')  
svm = joblib.load('svm_model.pkl')  
tree = joblib.load('dt_model.pkl')  
neural = joblib.load('nn_model.pkl')


# Initialize Flask app
app = Flask(__name__)

# Define route for prediction
@app.route("/")
def welcome():
    return render_template("homepage.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.get_json()
    input_features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = knn.predict(input_features)
    prediction = svm.predict(input_features)
    prediction = tree.predict(input_features)
    prediction = neural.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)