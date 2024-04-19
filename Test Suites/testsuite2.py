import unittest
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
import joblib
import re
import string
from Sarca import preprocess_text

class TestSarcasmDetector(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.dataset_path = "train.En.csv"
        self.df = pd.read_csv(self.dataset_path)
        
        # Preprocess the text data
        self.df = self.df.dropna()
        self.df['Text'] = self.df['Text'].apply(preprocess_text)
        
        # Split the dataset into features (X) and target variable (y)
        self.X = self.df['Text']
        self.y = self.df['Label']
        
        # Vectorize the text data using CountVectorizer
        self.vectorizer = CountVectorizer()
        self.X_vectorized = self.vectorizer.fit_transform(self.X)
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectorized, self.y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)
    
    def test_accuracy(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.5)  # Assuming at least 50% accuracy for a decent model
        
    def test_kappa(self):
        y_pred = self.model.predict(self.X_test)
        kappa = cohen_kappa_score(self.y_test, y_pred)
        self.assertGreaterEqual(kappa, 0)  # Kappa should be non-negative
        
    def test_f1_score(self):
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        self.assertGreaterEqual(f1, 0)  # F1 score should be non-negative
    
    def test_classification_report(self):
        y_pred = self.model.predict(self.X_test)
        classification_rep = classification_report(self.y_test, y_pred)
        self.assertTrue(len(classification_rep) > 0)  # Ensure classification report is not empty
    
    def test_model_saving(self):
        # Save the trained model
        model_filename = "sarcasm_detection_model.joblib"
        joblib.dump(self.model, model_filename)
        
        # Check if the model file exists
        import os
        self.assertTrue(os.path.exists(model_filename))
    
if __name__ == '__main__':
    unittest.main()
