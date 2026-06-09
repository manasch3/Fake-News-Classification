# Fake News Detection using NLP and Machine Learning

A Natural Language Processing (NLP) and Machine Learning project that classifies news articles as either **Fake** or **Real** using text preprocessing, TF-IDF feature extraction, and multiple supervised learning algorithms. The project explores different classification models, evaluates their performance, and identifies the most effective approach for fake news detection.

## Overview

The rapid spread of misinformation through online platforms has made fake news detection an important research problem. Manually verifying the authenticity of large volumes of news content is time-consuming and often impractical.

This project aims to automate the process of identifying fake news articles by leveraging Natural Language Processing and Machine Learning techniques. News articles are transformed into numerical feature vectors using TF-IDF and then evaluated using multiple classification algorithms to determine the most accurate model.

## Objectives

The primary objectives of this project are:

- Detect fake and real news articles automatically.
- Perform text preprocessing and feature extraction.
- Compare multiple machine learning models.
- Evaluate model performance using classification metrics.
- Analyze important words associated with fake and real news.
- Build an interactive prediction system for unseen news articles.

## Dataset

The project uses the Fake and Real News Dataset consisting of news articles collected from multiple online sources.

| Metric | Value |
|----------|----------|
| Dataset Type | Binary Classification |
| Classes | Fake News, Real News |
| Feature Used | News Article Text |
| Train/Test Split | 80/20 |
| Vectorization Method | TF-IDF |

### Dataset Source

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Methodology

The project follows a complete NLP pipeline beginning with text cleaning and ending with news authenticity prediction.

```text
News Articles
      ↓
Text Cleaning
      ↓
Stopword Removal
      ↓
Stemming
      ↓
TF-IDF Vectorization
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Fake / Real Prediction
```

### Text Preprocessing

To improve model performance, the textual data undergoes several preprocessing steps:

- Conversion to lowercase
- Removal of URLs
- Removal of special characters
- Stopword removal
- Word stemming using Porter Stemmer

These preprocessing techniques help reduce noise and improve feature quality.

### Feature Engineering

The cleaned news articles are converted into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).

TF-IDF captures the importance of words within documents while reducing the influence of extremely common words, making it highly effective for text classification tasks.

## Exploratory Data Analysis

Several visualizations were generated to better understand the dataset:

- Dataset distribution analysis
- Word count distribution
- Fake News Word Cloud
- Real News Word Cloud
- Accuracy comparison charts

These visualizations provide insights into the structure and characteristics of the news articles.

## Models Evaluated

Four machine learning algorithms were trained and evaluated.

| Model |
|---------|
| Logistic Regression |
| Naive Bayes |
| Random Forest |
| Support Vector Machine (SVM) |

The models were compared using standard classification metrics and accuracy scores to identify the best-performing classifier.

## Model Evaluation

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross Validation

The project also includes K-Fold Cross Validation to assess model stability and generalization performance.

## Feature Importance Analysis

One of the key aspects of this project is the analysis of words most strongly associated with fake and real news articles.

The model extracts:

- Top Fake News Keywords
- Top Real News Keywords

This provides additional interpretability and helps explain how the classifier makes decisions.

## Interactive News Prediction

The notebook includes an interactive prediction system where users can enter their own news article or headline and receive:

- Predicted Class (Fake or Real)
- Confidence Score

Example:

```text
Input:
Government announces new education policy reforms.

Prediction:
REAL NEWS

Confidence:
97.4%
```

## Repository Structure

```text
Fake-News-Classification/
│
├── Fake_News.ipynb
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
└── requirements.txt
```

## Running the Project

### Clone the Repository

```bash
git clone https://github.com/manasch3/Fake-News-Classification.git
cd Fake-News-Classification
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud joblib
```

### Launch the Notebook

```bash
jupyter notebook Fake_News.ipynb
```

The notebook includes data preprocessing, exploratory analysis, model training, evaluation, visualization, and interactive prediction.

## Key Findings

The project demonstrates that traditional machine learning algorithms combined with TF-IDF feature extraction can effectively distinguish between fake and real news articles.

Important observations include:

- NLP preprocessing significantly improves classification quality.
- TF-IDF provides strong text representations for fake news detection.
- Feature importance analysis helps interpret model decisions.
- Cross-validation confirms model stability.
- Machine learning can effectively automate news authenticity classification.

## Future Improvements

Potential future enhancements include:

- Fine-tuning transformer-based models such as BERT.
- Real-time news verification using web scraping.
- Explainable AI using SHAP and LIME.
- Multi-language fake news detection.
- Deployment as a web application using Flask or Streamlit.
