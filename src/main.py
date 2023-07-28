# Michael Jonathan Halim 13521124
# GAIB - Bonus - NLP

# Import libraries
import math
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define a simple Naive Bayes classifier
class NaiveBayesClassifier:
    def __init__(self):
        # Initialize data structures to store vocabulary, class probabilities, 
        # word counts, and word counts per class
        self.vocab = set()
        self.class_probabilities = {}
        self.word_counts = {}
        self.class_word_counts = {}

        # Load stopwords
        self.stop_words = pd.read_csv('stopwordbahasa.csv', header=None)
        self.stop_words = self.stop_words.rename(columns={0: 'stopword'})

        # Load alay dictionary
        self.alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
        self.alay_dict = self.alay_dict.rename(columns={0: 'original', 
                                            1: 'replacement'})
        self.alay_dict_map = dict(zip(self.alay_dict['original'], self.alay_dict['replacement']))

    def preprocess_text(self, text):
        # Lowercase text
        text = text.lower()

        # Remove unnecessary char
        # Remove every '\n'
        text = re.sub('\n',' ',text)

        # Remove every retweet symbol
        text = re.sub('rt',' ',text)

        # Remove every username
        text = re.sub('user',' ',text)

        # Remove every URL
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)

        # Remove extra spaces
        text = re.sub('  +', ' ', text)

        # Remove nonalphanumeric
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)

        # Normalize alay
        text = ' '.join([self.alay_dict_map[word] if word in self.alay_dict_map else word for word in text.split(' ')])
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter words from stop words
        words = [word for word in words if word not in self.stop_words]

        return words
    
    def fit(self, X_train, y_train):
        # Define total number of documents in the training set
        num_docs = len(X_train)

        # Define a set of unique classes in the training set
        self.classes = set(y_train)
        
        # Calculate class probabilities
        for cls in self.classes:
            self.class_probabilities[cls] = sum(y_train == cls) / num_docs
        
        # Build vocabulary and calculate word counts per class
        for x, y in zip(X_train, y_train):
            # Preprocess text to get individual words
            words = self.preprocess_text(x)

            # Update the vocabulary with unique words in the current document
            self.vocab.update(words)
            
            if y not in self.class_word_counts:
                # Initialize word counts dictionary for each class
                self.class_word_counts[y] = {}
            
            for word in words:
                # Increment word count for the current class
                if word not in self.class_word_counts[y]:
                    self.class_word_counts[y][word] = 0
                self.class_word_counts[y][word] += 1
        
        # Calculate word counts across all classes
        self.word_counts = {word: sum(self.class_word_counts[y].get(word, 0) for y in self.classes) for word in self.vocab}
    
    def predict(self, X_test):
        # Define predictions
        predictions = []

        # Iterate all test data
        for x in X_test:
            # Preprocess text to get individual words
            words = self.preprocess_text(x)
            scores = {cls: math.log(self.class_probabilities[cls]) for cls in self.classes}
            
            # Iterate all words in text
            for word in words:
                # If text not defined from training, skip
                if word not in self.vocab:
                    continue
                
                # Check for every class (0 and 1)
                for cls in self.classes:
                    # Get word count for the current class
                    word_count = self.class_word_counts[cls].get(word, 0)

                    # Total unique words in the current class
                    total_words = len(self.class_word_counts[cls])

                    # Laplace smoothing
                    smoothed_prob = (word_count + 1) / (self.word_counts[word] + total_words)

                    # Update scores
                    scores[cls] += math.log(smoothed_prob)
            
            # Select the class with the highest score as the prediction
            prediction = max(scores, key=scores.get)

            # Add the prediction to the list
            predictions.append(prediction)
        
        return predictions

if __name__ == "__main__":
    # Load data from CSV
    df = pd.read_csv("data.csv", encoding='latin-1')

    # Split data into X (text) and y (labels)
    X = df['Tweet']
    y = df['HS']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the NaiveBayesClassifier
    classifier = NaiveBayesClassifier()

    # Train the classifier on the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = classifier.predict(X_test)
    
    # Generate the classification report
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)
