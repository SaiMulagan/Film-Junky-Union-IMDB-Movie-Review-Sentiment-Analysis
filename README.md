# Film Junky Union: IMDB Movie Review Sentiment Analysis

This repository contains a machine learning project to classify IMDB movie reviews as positive or negative. The goal is to achieve an F1 score of at least 0.85.

## Project Description

The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. You'll be using a dataset of IMDB movie reviews with polarity labeling to build a model for classifying positive and negative reviews.

## Dataset

The dataset is stored in the `imdb_reviews.tsv` file and contains the following fields:
- `review`: the review text
- `pos`: the target, '0' for negative and '1' for positive
- `ds_part`: 'train'/'test' for the train/test part of the dataset

## Project Structure

1. **Initialization**
    ```python
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.corpus import stopwords
    import torch
    import re
    import transformers
    %matplotlib inline
    %config InlineBackend.figure_format = 'png'
    plt.style.use('seaborn')
    tqdm.pandas()
    ```

2. **Load Data**
    ```python
    df_reviews = pd.read_csv('datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})
    ```

3. **Exploratory Data Analysis (EDA)**
    ```python
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    
    ax = axs[0]
    dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates()['start_year'].value_counts().sort_index()
    dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
    dft1.plot(kind='bar', ax=ax)
    ax.set_title('Number of Movies Over Years')
    
    ax = axs[1]
    dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
    dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
    dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)
    dft2 = df_reviews['start_year'].value_counts().sort_index()
    dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
    dft3 = (dft2/dft1).fillna(0)
    axt = ax.twinx()
    dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)
    lines, labels = axt.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')
    ax.set_title('Number of Reviews Over Years')
    fig.tight_layout()

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    ax = axs[0]
    dft = df_reviews.groupby('tconst')['review'].count().value_counts().sort_index()
    dft.plot.bar(ax=ax)
    ax.set_title('Bar Plot of #Reviews Per Movie')
    
    ax = axs[1]
    dft = df_reviews.groupby('tconst')['review'].count()
    sns.kdeplot(dft, ax=ax)
    ax.set_title('KDE Plot of #Reviews Per Movie')
    fig.tight_layout()

    print("Dataset shape:", df_reviews.shape)
    print("First few rows:")
    print(df_reviews.head())
    print(df_reviews['pos'].value_counts())

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax = axs[0]
    dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
    dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
    dft.plot.bar(ax=ax)
    ax.set_ylim([0, 5000])
    ax.set_title('The train set: distribution of ratings')
    
    ax = axs[1]
    dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
    dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
    dft.plot.bar(ax=ax)
    ax.set_ylim([0, 5000])
    ax.set_title('The test set: distribution of ratings')
    fig.tight_layout()
    ```

4. **Preprocessing**
    ```python
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[\d\W]+', ' ', text)
        return text
    df_reviews['review_clean'] = df_reviews['review'].apply(preprocess_text)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    X = tfidf_vectorizer.fit_transform(df_reviews['review_clean'])
    y = df_reviews['pos'].astype(int)
    ```

5. **Train/Test Split**
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ```

6. **Modeling**
    ```python
    models = {
        'Dummy': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, proba)
        print(f'{name} F1 Score: {f1:.2f}')
        print(f'{name} ROC AUC: {roc_auc:.2f}')
        results[name] = {'F1': f1, 'ROC AUC': roc_auc}
    ```

7. **Evaluation**
    ```python
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        print(f'F1 Score: {f1:.2f}')
        print(f'ROC AUC: {roc_auc:.2f}')
    evaluate_model(models['Logistic Regression'], X_train, y_train, X_test, y_test)
    ```

8. **Visualization**
    ```python
    fpr, tpr, thresholds = roc_curve(y_test, models['Logistic Regression'].predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    ```

9. **Conclusion**
    The exploratory data analysis (EDA) revealed insights into the distribution of reviews and ratings over the years. Three models were implemented and compared: a Dummy Classifier, Logistic Regression, and Gradient Boosting Classifier. The Logistic Regression model achieved the highest performance with an F1 score of 0.90 and an ROC AUC of 0.96. The Gradient Boosting model had an F1 score of 0.82 and an ROC AUC of 0.89. The Dummy Classifier performed poorly as expected. Further improvements could involve parameter tuning and exploring additional advanced models like BERT for better performance.

## Instructions to Run the Project

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook or script:
    ```bash
    jupyter notebook Film_Junky_Union_IMDB_Movie_Review_Sentiment_Analysis.ipynb
    ```

## Future Work

- Improve data preprocessing and cleaning.
- Experiment with additional models and ensemble techniques.
- Implement and test BERT for a subset of the dataset for improved performance on GPU.

## Acknowledgements

The dataset was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

---

Feel free to contribute to this project by submitting a pull request or opening an issue for any bugs or improvements.
