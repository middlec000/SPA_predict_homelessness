# Licensing
Currently I have chosen to not license this repository. This means that no one besides myself has permission to copy, distribute, or modify this work. [More](https://choosealicense.com/no-permission/).

# Project Overview
## Objective
Predict the likelihood an individual will experience first-time homelessness based on their utility customer data.  

## Background
This project was the pilot study for Spokane Predictive Analytics (SPA), a collaboration between Avista Utilities, the City of Spokane, Urbanova, and Eastern Washington University. Data was provided by Avista Utilities and the City of Spokane. Matching and de-identifying (the process of replacing identifying information such as names and addresses with internally generated ID numbers) was performed by the data analysis team at Avista Utilities.   

This GitHub repository contains the code used to 1) investigate the data, 2) preprocess the data from the de-identified state into a useful format for model fitting, and 3) train models to predict if an individual will experience first-time homelessness based on their utility billing behavior.  

I used this project as my thesis for my MS in Applied Mathematics degree though Eastern Washington University. My thesis paper was accepted and I graduated in June, 2021.

## Data Description
The de-identified data was provided in three groups: geographical (GeoData_Anon.csv - unused), Avista service agreements (ServiceAgreements_Anon.csv), and billing data (SpaData_YYYY_Anon.csv).  
A complete list of all variables provided in all three groups can be found in the [data dictionary](supporting_documents/data_dictionary_md).  

The Geographical data was not used for this project because of an unfortunate tradeoff in granularity - the number of levels of (categorical) geographical identifiers was too large to predict on, but if these levels were aggregated they became unhelpful.  

The Billing data has Composite Key of (`SPA_ACCT_ID`, `SPA_PREM_ID`, `MONTH`) = (account id, location id, month) and consists of information related to customers missing payments and Avista's activity in seeking payment.  

The Service Agreements data has Composite Key (`SPA_ACCT_ID`, `SPA_PREM_ID`) and consists of information related to the types of service agreements the utility company has with each customer.  

After investigation the features deemed useful and used for model fitting were:  
| Feature           | Description                                                                                     | Type    | Source(s)                                                  |
|-------------------|-------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------|
| CMIS_MATCH        | Does the listed customer match an individual in the CMIS database? Matched using last 4 of SSN. | boolean | Avista                                                     |
| MONTH             | Month of arrears snapshot.                                                                      | integer | Avista/City of Spokane - utility payment                   |
| TOTAL_CUR_BALANCE | Total balance owed in gas, electric, water, sewer, and garbage utilities                        | float   | Avista - utility payment/City of Spokane – utility payment |
| BREAK_ARRANGEMENT | Start Severance: Break Arrangement                                                              | integer | Avista - collections activity                              |
| PAST_DUE          | Past Due Notice                                                                                 | integer | Avista - collections activity                              |
| SPA_PREM_ID       | Anonymized id mapping to an Avista premise.                                                     | integer | Avista                                                     |  

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
Supporting methods used for various programming tasks in this project.

## [preprocessing.py](Code/preprocessing.py)
### Billing
* Data is combined from multiple files.
* New feature created for combined amount owed in all utilities bills each month by a single customer account.
* Only relevant features retained.
* Dates are reformatted to number of months since December, 2015 (earliest month in dataset).
* All null values in amounts owed are dropped.
* Null values for `BREAK_ARRANGEMENT` and `PAST_DUE` are replaced with `0`s.
* There are a few duplicate (`SPA_ACCT_ID`, `SPA_PREM_ID`, `MONTH`)'s. Of these duplicates just take the last and drop the others.
### Service Agreements
* Data is loaded and features are renamed to match naming in billing data.
* Extraneous features are dropped.
* Some accounts have multiple people associated with them at one time, some only have one. Associate only the main account holder with each account and remove the other people. The main account holder is financially responsible for the account so the account activity is an indicator of the main account holder only.
* Replace null `CMIS_MATCH` values with `False`s.
* Update data so that if one instance of a person has `CMIS_MATCH` == `True`, then all instances of that person have `CMIS_MATCH` == `True`.
* Reformat dates to match billing data format (number of months since December, 2015).
* Find the earliest month a person was recorded experiencing homelessness and store it in `ENROLL_DATE`.
* Inner join to billing data on (`SPA_ACCT_ID`, `SPA_PREM_ID`).
### Combined
* There are more duplicate (`SPA_ACCT_ID`, `SPA_PREM_ID`, `MONTH`)'s. Of these duplicates just take the last and drop the others.
* Drop all data that occurs after an individual's `ENROLL_DATE`. This ensures we are predicting first-time (in this data) homelessness.
* Create feature `NUM_PREM_FOR_PER`, the cumulative number of premises a person has paid bills at each month.
* Create feature `NUM_PER_FOR_PREM`, the cumulative number of people a premises has seen for each month.

## [log_fit.py](Code/log_fit.py)
Using the method of K-Folds (k=4, splitting on `SPA_PER_ID`) the logistic model is fit to the data and predictions are made.  
Various performance metrics are calculated: tp, fp, tn, fn, tnr, ppv, npv, f-1 score, accuracy, balanced accuracy, area under the curve.  
Definitions and descriptions [here](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

## [plot_roc.py](Code/plot_roc.py)
The performance of the model is analyzed using a Receiver Operator Characteristic (ROC) plot.

## [main.py](Code/main.py)
Uses previous files to perform preprocessing, model fitting, and ROC curve plotting.

## [ROC_Curve.ipynb](Code/ROC_Curve.ipynb)
Another file for plotting the ROC Curve as well as comparing my model performance to current research and printing several tables used in my paper.