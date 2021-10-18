# Overview
## Objective
Predict the likelihood an individual will experience homelessness based on their utility customer data.  

## Background
This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration among: Avsita Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and deidentifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was perforemed by the data analysis team at Avsita Utilities.   

This Repository contains the code used to 1) investigate the data, 2) preprocess the data from this stage into a useful format for model fitting, and 3) train models to predict if an individual will experience homelessness based on thier utility billing behaviour.

# File Descriptions
## data_exploration
The data is explored using visual and numerical tools to answer several important questions about the data and relationships within.

## 0_preprocessing
The data is combined from multiple files, reformatted, and some new features are created in an effort to extract more information from the given data.

## 1_logistic_preprocess
The data is further manipulated to be used specifically for a logistic regression model.

## 1_ANN_preprocess
The data is further manipulated to be used specifically for a neural network model.

## 2_logistic
The logistic model is fit to the data and predictions are made.

## 2_LSTM
An Long-Short Term Memory model is fit to the data and predictions are made.*

## 2_Transformer
A Transformer model is fit to the data and predictions are made.*

## 3_ROC_curves
The predictions from each model are compared using a Receiver Operator Curve (ROC).

* In Progress