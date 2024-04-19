from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__, static_url_path='/static')



app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")
def index():
    return render_template('homepage.html')

#Load the trained model
model = joblib.load("sarcasm_detection_model.joblib")

#Preprocess the text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

#Vectorizer for transforming text data
vectorizer = CountVectorizer()

@app.route("/detect-sarcasm", methods=['POST'])
def detect_sarcasm():
    data = request.get_json()
    user_input = data['text']

    #Preprocess user input and transform using vectorizer
    processed_input = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([processed_input])

    #Make prediction using the model
    prediction = model.predict(input_vectorized)[0]

    result = {'isSarcastic': bool(prediction)}

    return jsonify(result)






if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=8000, threads=4)
     app.run(host='127.0.0.1', port=5000)
    