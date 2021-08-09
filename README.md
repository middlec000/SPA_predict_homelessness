# Overview
Objective: predict homelessness based on customer utility billing data.  

This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration among: Avsita Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and deidentifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was perforemed by the data analysis team at Avsita Utilities.   

This Repo contains the code used to 1) preprocess the data from this stage, 2) investigate the data, and 3) train models to predict if an individual will experience homelessness based on thier utility billing behaviour.

# Source Code Files
## 0_preprocessing
The data is combined from multiple files, preprocessed, and some new features are created in an effort to extract more information from the data.

## 1_logistic_preprocess
The data is further manipulated to be used specifically for logistic regression.

## 1_ANN_preprocess
The data is further manipulated to be used specifically for a neural network.

## 2_logistic
The logistic model is fit to the data.

## 2_ANN
A neural network is fit to the data.

## 3_ROC_curves
The model predictions are compared using a Receiver Operator Curve (ROC).

## data_exploration
The data is explored to answer several important questions about the data and relationships within.
