# Sentiment Analysis of IMDB Movie Reviews
This project focuses on building a machine learning model for sentiment analysis of movie reviews. It explores the use of Natural Language Processing (NLP) techniques to classify reviews as positive or negative. The project is implemented in Python using the scikit-learn library and evaluates three different models: Logistic Regression, Naive Bayes, and Linear Support Vector Classification (SVC).

## Project Overview:
### Problem Statement: 
Classify movie reviews as either positive or negative sentiment based on their text content.
### Dataset: 
Utilizes the IMDB movie review dataset, containing 50,000 movie reviews labeled as positive or negative.
### Goal: 
Develop and compare the performance of different machine learning models for sentiment classification.

## Methodology
### Data Preprocessing:

Removed HTML tags from the reviews.
Removed non-alphabetic characters and converted text to lowercase.
Tokenized the reviews into individual words.
Removed common English stop words (e.g., "the," "a," "an").
Applied stemming to reduce words to their base forms (e.g., "loved" to "love").

### Feature Extraction:
Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to transform text into numerical features.
Limited the maximum number of features to 5000 to reduce dimensionality.

### Model Selection and Training:

Evaluated three models: Logistic Regression, Naive Bayes, and Linear SVC.
Trained each model on the preprocessed data.

### Model Evaluation:
Assessed model performance on a test set (20% of the data) using:
Accuracy: Percentage of correct predictions.
Precision: Accuracy of positive predictions.
Recall: Ability to find all positive instances.
F1-score: Harmonic mean of precision and recall.
Confusion matrix: Visualization of prediction errors.

### Results

| Model             | Accuracy | Precision | Recall | F1-Score |
|-------------------|:-------:|:--------:|:-----:|:-------:|
| Logistic Regression | 0.8814   | 0.87      | 0.89   | 0.88    |
| Naive Bayes      | 0.8814   | 0.87      | 0.89   | 0.88    |
| Linear SVC        | 0.8814   | 0.87      | 0.89   | 0.88    |


All three models demonstrated similar performance, achieving an accuracy of 88.14%. This indicates that they can effectively classify movie reviews as positive or negative. The confusion matrices also reveal a balanced performance in identifying both positive and negative sentiments.

## Conclusion
This project successfully showcases the implementation of sentiment analysis on movie reviews using various machine learning models. The results demonstrate that text preprocessing and TF-IDF vectorization are effective techniques for preparing text data for sentiment analysis tasks.

## Future Work
### Hyperparameter Tuning: 
Optimize model hyperparameters for potentially improved performance.
### Ensemble Methods: 
Explore combining multiple models to potentially achieve even better accuracy.
### Deep Learning Models: 
Experiment with more advanced neural network architectures, such as Recurrent Neural Networks (RNNs) or Transformers, for potentially higher accuracy.
### Feature Engineering: 
Investigate more sophisticated feature extraction techniques, such as word embeddings, to capture nuanced semantic information.
