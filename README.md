# Licensing
Currently I have chosen to not license this repository. This means that no one besides myself has permission to copy, distribute, or modify this work. [More](https://choosealicense.com/no-permission/).

# Overview
## Objective
Predict the likelihood an individual will experience homelessness based on their utility customer data.  

## Background
This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration among: Avista Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and de-identifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was performed by the data analysis team at Avista Utilities.   

This Repository contains the code used to 1) investigate the data, 2) preprocess the data from this stage into a useful format for model fitting, and 3) train models to predict if an individual will experience homelessness based on their utility billing behavior.

## Data Description
The data was provided in three groups: geographical (GeoData_Anon - unused), Avista service agreements (ServiceAgreements_Anon), and billing data (SpaData_YYYY_Anon). A complete list of all variables provided in all three groups can be found in the [data dictionary](supporting_documents/data_dictionary_md).  

The geographical data was not used for this project because of the tradeoff of granularity - the number of levels of (categorical) geographical identifiers was too large to predict on, but if these levels were further grouped they became less useful.  

The billing data has Composite Key of ('SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH') = (account id, location id, month) and consists of information related to customers missing payments and Avista's activity in seeking payment.

The service agreements data has Composite Key ('SPA_ACCT_ID', 'SPA_PREM_ID') and consists of information related to the types of service agreements the utility company has with each customer.  

After preprocessing the data used consists of the following:  


# File Descriptions in Code/
## [0_Data_Exploration.ipynb](Code/0_Data_Exploration.ipynb)
The data is explored using visual and numerical tools to answer several important questions about the data and investigate relationships within.
### Which Outcome Measure to Use?
Determine which of several potential outcome measures is the most correlated with the provided data.
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

## [helper_methods.py](Code/helper_methods.py)
Supporting methods used by other files.

## [preprocessing.py](Code/preprocessing.py)
The data is combined from multiple files, reformatted, and some new features are created in an effort to extract more information from the given data.

## [log_fit.py](Code/log_fit.py)
The logistic model is fit to the data produced from `1_Logistic_Preprocess` and predictions are made.

## [plot_roc.py](Code/plot_roc.py)
The predictions from each model are compared using a Receiver Operator Characteristic (ROC) plot.

## [main.py](Code/main.py)
Uses previous files to perform preprocessing, model fitting, and ROC curve plotting.

## [ROC_Curve.ipynb](Code/ROC_Curve.ipynb)
Another way to plot the ROC Curve as well as print several tables used in the paper.