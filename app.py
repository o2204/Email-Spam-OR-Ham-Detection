# Import necessary libraries
from flask import Flask, render_template, request
from preprocessing import data_preprocessing
import joblib



# Load the pre-trained model and vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')

# Initialize Flask application
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['message']

    cleaned_email = data_preprocessing(email)
    vect = tfidf_vectorizer.transform([cleaned_email])
    prediction = svm_model.predict(vect)
    label = encoder.inverse_transform(prediction)[0]

    prediction_value = 1 if label.lower() == 'spam' else 0
    return render_template('result.html', prediction=prediction_value)

if __name__ == "__main__":
    app.run(debug=True) 
