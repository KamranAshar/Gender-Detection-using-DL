Gender Detection using DL:-
Gender detection using deep learning involves building and training a neural network on a labeled dataset to classify gender based on input features. With a properly collected and preprocessed dataset, and a well-chosen model architecture, deep learning can achieve high accuracy in gender classification tasks.

Libraries Used:
- nltk
- random
- DictVectorizer from sklearn.feature_extraction
- MultinomialNB from sklearn.naive_bayes
- Pipeline from sklearn.pipeline
- train_test_split from sklearn.model_selection
- accuracy_score from sklearn.metrics
- classification_report from sklearn.metrics

Functions Defined:
- gender_features
- predict_gender

Step 1: Import Libraries and Download NLTK Data.
- Imports necessary libraries.
- Downloads the 'names' dataset from the NLTK library if it's not already downloaded.

Step 2: Define Feature Extraction Function.
- Defines a function gender_features that extracts specific features from a name such as the last letter, last two letters, last three letters, first letter, first two letters, and first three letters. These features are used for training the classifier.

Step 3: Load and Shuffle Dataset.
- Loads the dataset of male and female names from the NLTK library.
- Combines the male and female names into a single list labeled_names with labels ('male' or 'female').
- Shuffles the combined dataset to ensure that the data is randomly distributed.

Step 4: Extract Features and Labels, Split Dataset.
- Extracts features for each name using the gender_features function and stores them in features.
- Extracts the corresponding labels ('male' or 'female') and stores them in labels.
- Splits the dataset into training and testing sets using an 80-20 split.

Step 5: Create and Train the Model.
- Creates a pipeline with two steps:
- DictVectorizer to convert the dictionary of features into a format suitable for the classifier.
- MultinomialNB to create a Naive Bayes classifier.

Step 6: Evaluate the Model.
- Uses the trained model to make predictions on the test set.
- Calculates the accuracy of the model and prints a detailed classification report showing precision, recall, and F1-score for each class ('male' and 'female').

Step 7:  Define Prediction Function and Test It.
- Defines a function predict_gender that takes a name as input, extracts its features, and uses the trained model to predict its gender.
- Tests the predict_gender function with example names 'John' and 'Mary'.
