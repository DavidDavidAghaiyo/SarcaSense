import unittest
from Sarca import preprocess_text

class TestPreprocessText(unittest.TestCase):
    def test_lowercase_conversion(self):
        text = "This is a Test STRING"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "this is a test string")

    def test_remove_numbers(self):
        text = "This is 123 a Test 456 STRING"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "this is  a test  string")

    def test_remove_punctuation(self):
        text = "This is a Test, STRING! With. Punctuation?"
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "this is a test string with punctuation")

    def test_strip_whitespace(self):
        text = "   This is a Test STRING    "
        processed_text = preprocess_text(text)
        self.assertEqual(processed_text, "this is a test string")

if __name__ == '__main__':
    unittest.main()
