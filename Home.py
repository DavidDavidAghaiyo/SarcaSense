import joblib
import re
import string
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import gradio as gr

# Load the dataset for vectorizer fitting
dataset_path = "train.En.csv"
df = pd.read_csv(dataset_path)

# Preprocess the text data for vectorizer fitting
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = text.strip()  # Remove leading and trailing whitespaces
    else:
        text = ''  # Replace NaN values with an empty string
    return text

# Fit CountVectorizer on preprocessed data
vectorizer = CountVectorizer()
df['Text'] = df['Text'].apply(preprocess_text)
X_vectorized = vectorizer.fit_transform(df['Text'])

# Save the CountVectorizer object
vectorizer_filename = "count_vectorizer.joblib"
joblib.dump(vectorizer, vectorizer_filename)

# Train and save the model
X = X_vectorized
y = df['Label']
model = MultinomialNB()
model.fit(X, y)

model_filename = "sarcasm_detection_model.joblib"
joblib.dump(model, model_filename)

print(f"Model and CountVectorizer saved as {model_filename} and {vectorizer_filename}")

# Load the trained model and CountVectorizer
model_filename = "sarcasm_detection_model.joblib"
model = joblib.load(model_filename)

vectorizer_filename = "count_vectorizer.joblib"
vectorizer = joblib.load(vectorizer_filename)

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
    else:
        text = ''
    return text

def welcome_message():
    welcome_text = "Welcome to SarcaSense"
    for letter in welcome_text:
        time.sleep(1)
        yield letter

# Function to predict sarcasm based on input text
def predict_sarcasm(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return "Sarcastic 🙁" if prediction == 1 else "Not Sarcastic 🙂"


# Custom theme with primary hue as blue and secondary hue as green
custom_theme = gr.themes.Default(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.green)

# Custom CSS styling for the interface
custom_css = ".gradio-container {background-color: #f0f0f0; border-radius: 10px;}"

# Create Gradio interface with custom theme and CSS
iface = gr.Interface(predict_sarcasm, inputs="text", outputs="text", title="Welcome to SarcaSense", description="Enter a sentence to detect sarcasm.", theme=custom_theme, css=custom_css)

# Display 'Welcome to SarcaSense' with letters appearing one after the other on first run
iface.launch(inline=True, debug=True)
