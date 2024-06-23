#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries.
# Downloading the 'names' dataset from the NLTK library if it's not already downloaded.

# In[3]:


# Import necessary libraries
import nltk
from nltk.corpus import names
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure that the required NLTK datasets are downloaded
nltk.download('names')


# # Defining Feature Extraction Function
# Defining a function gender_features that extracts specific features from a name such as the last letter, last two letters, last three letters, first letter, first two letters, and first three letters. These features are used for training the classifier.

# In[6]:


# Define a function to extract features from names
def gender_features(word):
    """
    Extracts features from a given name.
    
    Args:
        word (str): The name from which to extract features.
        
    Returns:
        dict: A dictionary containing features of the name.
    """
    return {
        'last_letter': word[-1],
        'last_two_letters': word[-2:],
        'last_three_letters': word[-3:],
        'first_letter': word[0],
        'first_two_letters': word[:2],
        'first_three_letters': word[:3]
    }


# # Loading and Shuffling Dataset
# Loads the dataset of male and female names from the NLTK library.
# Combines the male and female names into a single list labeled_names with labels ('male' or 'female').
# Shuffles the combined dataset to ensure that the data is randomly distributed.

# In[7]:


# Load the dataset
male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names

# Shuffle the dataset
random.shuffle(labeled_names)


# # Extracting Features and Labels, Split Dataset
# Extracts features for each name using the gender_features function and stores them in features.
# Extracts the corresponding labels ('male' or 'female') and stores them in labels.
# Splits the dataset into training and testing sets using an 80-20 split.

# In[8]:


# Extract features and labels from the dataset
features = [gender_features(name) for name, gender in labeled_names]
labels = [gender for name, gender in labeled_names]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# # Creating and Training the Model
# Creating a pipeline with two steps:
# DictVectorizer to convert the dictionary of features into a format suitable for the classifier.
# MultinomialNB to create a Naive Bayes classifier.
# Training the model using the training data.

# In[10]:


# Create a pipeline with a DictVectorizer and a Multinomial Naive Bayes classifier
model = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)


# # Evaluating the Model
# Uses the trained model to make predictions on the test set.
# Calculates the accuracy of the model and prints a detailed classification report showing precision, recall, and F1-score for each class ('male' and 'female').

# In[12]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


# # Define Prediction Function and Test It
# Defining a function predict_gender that takes a name as input, extracts its features, and uses the trained model to predict its gender.
# Testing the predict_gender function with example names 'John' and 'Mary'.

# In[15]:


# Define a function to predict the gender of a given name
def predict_gender(name):
    """
    Predicts the gender of a given name.
    
    Args:
        name (str): The name to predict the gender for.
        
    Returns:
        str: The predicted gender ('male' or 'female').
    """
    features = gender_features(name)
    return model.predict([features])[0]

# Test the prediction function with example names
print(predict_gender('Mary'))  # Expected output: 'male'
print(predict_gender('John'))  # Expected output: 'female'

