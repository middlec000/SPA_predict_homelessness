# Licensing
Currently I have chosen to not license this repository. This means that no one besides myself has permission to copy, distribute, or modify this work. [More](https://choosealicense.com/no-permission/).

# Overview
## Objective
Predict the likelihood an individual will experience homelessness based on their utility customer data.  

## Background
This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration among: Avsita Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and deidentifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was perforemed by the data analysis team at Avsita Utilities.   

This Repository contains the code used to 1) investigate the data, 2) preprocess the data from this stage into a useful format for model fitting, and 3) train models to predict if an individual will experience homelessness based on thier utility billing behaviour.

# File Descriptions
## 00_Data_Exploration
The data is explored using visual and numerical tools to answer several important questions about the data and investigate relationships within.
### Which Outcome Measure to Use?
Determine which of several potential outcome measures is the most correlated with the predictor attributes.
### Which Years to Use?
Determine if there is a difference in relationships between predictors and outcome in different years.
### Which Billing Attributes to Keep?
Determine which level of aggregation of customer billing attributes is the most correlated with the outcome measure.
### Which Other Attributes to Use?
Determine which attributes besides the billing to include in model fitting.
### Data Imbalance
Determine the degree of imbalance in the data.
### Time and Events
Analyze when events occur over time and for what months we have data on them.
### Compare P and N
Look at the attributes most correlated with the outcome and assess how different the distributions of positives and negatives appear on each attribute.
### Geographical
Determine if there are any useful groupings of positives or negatives based on geographical attributes.

## 0_Preprocessing
The data is combined from multiple files, reformatted, and some new features are created in an effort to extract more information from the given data.

## 1_ANN_Preprocess
The data is further manipulated from the output of `0_Preprocessing` to be used specifically for a neural network model.

## 1_Logistic_Preprocess
The data is further manipulated from the output of `0_Preprocessing` to be used specifically for a logistic regression model.

## 2_ANN_Fit \*
A vanilla neural network model is fit to the data produced from `1_ANN_Preprocess` and predictions are made.

## 2_LSTM_Fit \*
An Long-Short Term Memory model is fit to the data produced from `1_ANN_Preprocess` and predictions are made.

## 2_Logistic_Fit
The logistic model is fit to the data produced from `1_Logistic_Preprocess` and predictions are made.

## 2_Transformer_Fit \*
A transformer model is fit to the data produced from `1_ANN_Preprocess` and predictions are made.

## 3_Logistic_Postprocessing
Take output from `2_Logistic_Fit` and take the maximum risk for each person. Also reformat for use in `4_Indicative_Time`.

## 4_Indicative_Time
The predictions from the logistic regression model are analyzed to determine if there are one or more common high predicted risk times.

## 4_ROC_Curves
The predictions from each model are compared using a Receiver Operator Characteristic (ROC) plot.

\* In Progress or Unfinished