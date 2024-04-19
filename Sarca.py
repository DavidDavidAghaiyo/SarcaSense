import joblib
import re 
import string

import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

#Load the dataset 
dataset_path = "train.En.csv"
df = pd.read_csv(dataset_path)

#Preprocess the text data
def preprocess_text(text):
    text = text.lower() #Convert to lowercase
    text = re.sub(r'\d+', '', text) #Remove numbers
    text = text.translate(str.maketrans('','', string.punctuation)) #Remove punctuation
    text = text.strip() #Remove leading and trailing whitespaces 
    return text 
#Remove missing values
df= df.dropna()
df['Text'] = df['Text'].apply(preprocess_text)

#Split the dataset into features (X) and target variable (y)
X = df['Text']
y = df['Label'] #Label being the column containing the target variable

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#Initialize the Naive Bayes model
model = MultinomialNB()

#Train the model
model.fit(X_train_vectorized, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
classification_rep = classification_report(y_test, y_pred)

#Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Cohen's Kappa: {kappa}")
print(f"F1 Score: {f1}")
print("Classification Report: \n", classification_rep)

# Save the trained model
model_filename = "sarcasm_detection_model.joblib"
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}")
