# Fake Review Detector

Fake Review Detector is a machine learning-based application designed to identify whether a product review is genuine or fake, helping improve trust in online platforms.

## Project Overview

This project uses Natural Language Processing (NLP) and a supervised machine learning model to classify reviews as real or fake. The system is trained on a dataset containing both authentic and computer-generated reviews and learns patterns in writing style and word usage.

## Technologies Used

- Python  
- Scikit-learn  
- Natural Language Processing (NLP)  
- TF-IDF Vectorization  
- Logistic Regression  
- Streamlit (for deployment)  
- GitHub (for version control)  

## Key Features

- Detects fake and genuine product reviews  
- Uses TF-IDF to convert text into numerical features  
- Applies Logistic Regression for classification  
- Provides instant predictions through a simple interface  
- Lightweight and efficient model suitable for real-time use  

## Workflow

- The dataset is loaded and preprocessed by cleaning and normalizing text  
- Stopwords are removed to focus on meaningful words  
- Text data is converted into numerical format using TF-IDF vectorization  
- A Logistic Regression model is trained on the processed data  
- The trained model is used to predict whether new reviews are fake or real  
- The application displays results through a Streamlit web interface  

## Demo

The application provides a simple interface where users can input a product review and receive a prediction indicating whether the review is fake or genuine.
URL- https://fake-review-detector-hxszbdz4y5qbsjpebblgzv.streamlit.app/

## Status

Project completed and deployed for demonstration.
