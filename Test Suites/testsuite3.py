import unittest
import os
import joblib
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
import string

class TestSarcasmDetectorRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset
        dataset_path = "train.En.csv"
        df = pd.read_csv(dataset_path)
        
        # Preprocess the text data
        def preprocess_text(text):
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            text = text.strip()  # Remove leading and trailing whitespaces
            return text
        
        df = df.dropna()
        df['Text'] = df['Text'].apply(preprocess_text)
        
        # Split the dataset into features (X) and target variable (y)
        X = df['Text']
        y = df['Label']
        
        # Split the data into training and testing sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize the text data using CountVectorizer
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(cls.X_train)
        X_test_vectorized = vectorizer.transform(cls.X_test)
        
        # Initialize and train the model
        model = MultinomialNB()
        model.fit(X_train_vectorized, cls.y_train)
        
        # Save the trained model
        cls.model_filename = "sarcasm_detection_model_regression_test.joblib"
        joblib.dump(model, cls.model_filename)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate evaluation metrics
        cls.accuracy = accuracy_score(cls.y_test, y_pred)
        cls.kappa = cohen_kappa_score(cls.y_test, y_pred)
        cls.f1 = f1_score(cls.y_test, y_pred, average='weighted')
        cls.classification_rep = classification_report(cls.y_test, y_pred)
    
    def test_model_performance(self):
        # Load the saved model
        saved_model = joblib.load(self.model_filename)
        
        # Check if the loaded model is not None
        self.assertIsNotNone(saved_model)
        
        # Vectorize the test data
        vectorizer = CountVectorizer()
        X_test_vectorized = vectorizer.transform(self.X_test)
        
        # Make predictions using the loaded model
        y_pred = saved_model.predict(X_test_vectorized)
        
        # Check if the evaluation metrics remain consistent
        self.assertAlmostEqual(accuracy_score(self.y_test, y_pred), self.accuracy, delta=0.01)
        self.assertAlmostEqual(cohen_kappa_score(self.y_test, y_pred), self.kappa, delta=0.01)
        self.assertAlmostEqual(f1_score(self.y_test, y_pred, average='weighted'), self.f1, delta=0.01)
        self.assertEqual(classification_report(self.y_test, y_pred), self.classification_rep)
        
        # Delete the model file after testing
        os.remove(self.model_filename)

if __name__ == '__main__':
    unittest.main()
