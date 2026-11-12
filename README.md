
# Customer Review Sentiment Analysis using Naive Bayes

## Project Overview

This project focuses on building a Machine Learning application that classifies customer reviews into Positive and Negative sentiments using Naive Bayes classification algorithms.

The application uses Natural Language Processing (NLP) techniques to preprocess customer reviews, convert text into numerical features, train classification models, and predict the sentiment of new reviews.

## Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess customer review text.
- Convert text into numerical features using Bag of Words and TF-IDF.
- Train Naive Bayes classification models.
- Evaluate model performance using classification metrics.
- Build an interactive web application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Streamlit
- Joblib

## Project Workflow

1. Data Loading and Exploration
2. Text Preprocessing
3. Train-Test Split
4. Feature Extraction
   - CountVectorizer (Bag of Words)
   - TF-IDF Vectorizer
5. Model Training
   - Bernoulli Naive Bayes
   - Multinomial Naive Bayes
6. Model Evaluation
7. Sentiment Prediction
8. Streamlit Deployment

## Machine Learning Models

### 1. Bernoulli Naive Bayes
Uses binary word-presence features generated through Bag of Words.

### 2. Multinomial Naive Bayes
Uses TF-IDF numerical features for sentiment classification.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

## Project Structure

```text
Naive_Bayes_Sentiment_Analysis/
│
├── models/
│   ├── bernoulli_model.pkl
│   ├── bow_vectorizer.pkl
│   ├── multinomial_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── sentiment_analysis.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Naive_Bayes_Sentiment_Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

## Sample Reviews

| Customer Review | Expected Sentiment |
|---|---|
| The food was fantastic! | Positive |
| Worst service ever. | Negative |
| I really enjoyed the experience. | Positive |
| The product quality is very poor. | Negative |

## Future Enhancements

- Improve model performance through hyperparameter tuning.
- Explore additional machine learning algorithms.
- Add more customer review datasets.
- Improve the Streamlit interface.
- Deploy the application publicly.

## Author

Yuvaraj Naredla

## Acknowledgement

This project was developed as part of Machine Learning and Natural Language Processing practice.
