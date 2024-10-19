# **Women Cloth Reviews Prediction with Multinomial Naïve Bayes**

-------------

## **Objective**
The goal of this project is to predict the sentiment of women's clothing reviews using the Multinomial Naïve Bayes algorithm. We aim to classify reviews as either positive or negative based on the text of the review.

## **Data Source**
The dataset used for this project can be obtained from [Kaggle](https://www.kaggle.com/datasets) or any other suitable source where women's clothing reviews are available. Ensure to specify the exact dataset used.

## **Import Library**
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv('women_clothing_reviews.csv')  # Adjust the filename as necessary

# Explore the dataset
print(data.head())
print(data.describe())
print(data.info())
print(data['Review'].value_counts())  # Assuming 'Review' is the text column

# Visualize the distribution of sentiments
sns.countplot(x='Sentiment', data=data)  # Assuming 'Sentiment' is the target column
plt.title('Distribution of Review Sentiments')
plt.show()

# Handle missing values
data.dropna(subset=['Review', 'Sentiment'], inplace=True)  # Adjust as necessary

# Encode the sentiment labels
data['Sentiment'] = data['Sentiment'].map({'positive': 1, 'negative': 0})  # Example encoding

X = data['Review']  # Features (text reviews)
y = data['Sentiment']  # Target variable (sentiment)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Fit a Multinomial Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate the model's performance
predictions = model.predict(X_test_vectorized)
print('Accuracy:', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Make predictions on new reviews
new_reviews = pd.Series(["I love this dress!", "This is the worst purchase ever!"])  # Example new reviews
new_reviews_vectorized = vectorizer.transform(new_reviews)
new_predictions = model.predict(new_reviews_vectorized)

# Output predictions
for review, prediction in zip(new_reviews, new_predictions):
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f'Review: "{review}" - Sentiment: {sentiment}')
