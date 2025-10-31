# hate-speech-classification-ml
# Hate Speech Detection with TF-IDF and Stacked Ensemble Models
**Kaggle Competition Ranking:** 9th out of 36 teams (F1-score: 0.71293)

---

## Overview

This project aims to detect hate speech in social media posts using a **machine learning pipeline** built in Python. It focuses on building **high-performing models** through feature engineering, stacked ensemble learning, and efficient processing techniques to handle large-scale text data.

---

## Problem Statement

Detecting hate speech online is crucial for reducing harmful content and creating safer social media environments. The challenge involves:

- Large datasets with noisy, unstructured text  
- Subtle context differences, e.g., "not bad" vs "bad"  
- Need for scalable solutions for high-dimensional feature sets  

---

## Data Processing & Feature Engineering

We implemented **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction with the following enhancements:

- **Unigrams & Bigrams:** Capture single words and two-word combinations to retain context.  
- **Max Features = 70,000:** Focus on the most relevant terms and reduce overfitting.  
- **Parallel Processing:** Speeds up TF-IDF transformations using `joblib`.  
- **Dask:** Handles large datasets exceeding memory limits by processing in chunks.  
- **Preprocessing Experiments:** Explored stop-word removal, lemmatization, and stemming. The best results were achieved without these.  

---

## Modeling

The project uses a **stacked ensemble approach** combining:

- **CatBoost Classifier** – captures non-linear patterns efficiently  
- **Logistic Regression** – handles sparse high-dimensional data  
- **Decision Tree** – as the final estimator to combine base predictions  

We also experimented with hyperparameter tuning using **Grid Search** and **Random Search** to optimize performance.

---

## Stacked Ensemble Approach

| Base Models         | Final Estimator        | F1 Score  |
|--------------------|----------------------|-----------|
| CatBoost + LR      | Decision Tree         | 0.70874   |
| CatBoost + LR      | Logistic Regression   | 0.70592   |
| LR + CNB + CatBoost| Decision Tree         | 0.70595   |

Key Highlights:  

- The stacking method effectively combines predictions from diverse models.  
- Dask ensures memory efficiency while handling large feature sets (~70,000 features).  
- Parallelized TF-IDF computation reduces processing time.  

---

## Performance

- **F1-Score:** 0.71293 (9th out of 36 teams on Kaggle)  
- Efficient pipeline capable of handling large-scale textual data.  
- Flexible and scalable for additional feature engineering or model updates.  

---

## Technologies Used

- **Python:** Core language for data processing and modeling  
- **pandas & Dask:** Data loading, cleaning, and chunked processing  
- **scikit-learn:** Machine learning pipelines and ensemble methods  
- **CatBoost:** Gradient boosting classifier for high-dimensional data  
- **Joblib:** Parallelized computation for faster TF-IDF transformations  
- **F1-score evaluation:** Metric for model performance  

---
