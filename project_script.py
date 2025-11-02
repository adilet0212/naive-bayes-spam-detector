# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 14:22:31 2023

@author: Adilet
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data into a pandas DataFrame
data = pd.read_csv(r'C:\Users\Adilet\Downloads\Youtube01-Psy.csv')

# Display the first few rows of the DataFrame
data.head()

# Basic Data Exploration
# Basic information about the dataset
data_info = data.info()

# Descriptive statistics for the numerical columns
data_description = data.describe()

# Check for missing values
missing_values = data.isnull().sum()

# Distribution of the class labels
class_distribution = data['CLASS'].value_counts()

# A few examples of comments
sample_comments = data['CONTENT'].sample(5)

# Printing all of them
data_info, data_description, missing_values, class_distribution, sample_comments

# Selecting the Content column for text data and Class column as labels
content = data['CONTENT']
labels = data['CLASS']

# Initialize CountVectorizer
# As we know, NLTK's English stopwords are used to filter out common words
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

# Transform the text data to a matrix of token counts
content_transformed = vectorizer.fit_transform(content)

# Displaying the shape of the transformed data
print("Shape of the transformed data:", content_transformed.shape)

# Displaying some feature names
print("Some feature names:", vectorizer.get_feature_names_out()[:10])

# Initializing the TF-IDF transformer to convert word counts to TF-IDF scores
tfidf_transformer = TfidfTransformer()

# Apply TF-IDF transformation
content_tfidf = tfidf_transformer.fit_transform(content_transformed)

# Displaying the shape of the TF-IDF transformed data
print("Shape of TF-IDF transformed data:", content_tfidf.shape)

# Convert the TF-IDF data to a DataFrame
tfidf_df = pd.DataFrame(content_tfidf.toarray())

# Add the labels as a new column
tfidf_df['LABEL'] = labels

# Shuffle the DataFrame
shuffled_df = tfidf_df.sample(frac=1, random_state=42)

# Separate the features and labels
shuffled_features = shuffled_df.drop('LABEL', axis=1)
shuffled_labels = shuffled_df['LABEL']

# Calculate the number of samples for the training set (75% of the dataset)
train_size = int(len(shuffled_df) * 0.75)

# Split the features
train_features = shuffled_features.iloc[:train_size]
test_features = shuffled_features.iloc[train_size:]

# Split the labels
train_labels = shuffled_labels.iloc[:train_size]
test_labels = shuffled_labels.iloc[train_size:]

# Print the shapes to confirm the split
print("Training features shape:", train_features.shape)
print("Testing features shape:", test_features.shape)
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)

# Initialize the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(train_features, train_labels)

# Perform 5-fold cross-validation on the training data
cv_scores = cross_val_score(nb_classifier, train_features, train_labels, cv=5)

# Calculate and print the mean of the cross-validation scores
mean_cv_score = cv_scores.mean()
print("Mean CV Score:", mean_cv_score)

# Predicting the labels for the test set
predicted_labels = nb_classifier.predict(test_features)

# Calculating the accuracy of the model on the test set
test_accuracy = accuracy_score(test_labels, predicted_labels)

# Generating the confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Displaying the accuracy and confusion matrix
print("Test Accuracy:", test_accuracy)
print("Confusion Matrix:\n", conf_matrix)

# New comments for testing
new_comments = [
    "Your current bank account balance is $15250.45",  # Non-spam example
    "I love cars. My favourite ones are BMW and Mercedes.",  # Non-spam example
    "How can you lose weight in one month? Check out our product!",  # Spam example
    "Fantastic video, it was very informative and helpful.",  # Non-spam example
    "Win a new phone by clicking this link now!",  # Spam example
    "Nowadays, I think technology is advancing tremendously!"  # Non-spam example
]

# Transforming the new comments using the same vectorizer and tfidf transformer
new_comments_transformed = vectorizer.transform(new_comments)
new_comments_tfidf = tfidf_transformer.transform(new_comments_transformed)

# Predicting the class of the new comments
new_comments_predictions = nb_classifier.predict(new_comments_tfidf)

# Displaying both the comments and predictions (0 - non-spam, 1 - spam)
for comment, prediction in zip(new_comments, new_comments_predictions):
    print(f'Comment: "{comment}"\nPredicted Class: {prediction}\n')



