# Changes to Paper
## New Outline
### Introduction  
* Predicting homelessness on large scale (state, national) preferable
    - Lose tracking of fewer people since they all stay in same larger region
    - Additional information sources when people move
* Other studies' data sources
    - Medical (emergency services)
    - Mental health
    - Financial assistance usage
    - Department of Justice
* Really want financial info
    - Potential sources: taxes, salary, etc.
    - We use utility billing history as proxy

### Methods  
* Data
    - Sources
    - Preparation
    - Feature selection
* Model
    - Logistic on person-place-month
    - maximum risk for person
* Evaluation
    - K-Folds
    - ROC curve

### Results  
* ROC curve
* Other metrics comparable to other studies: Precision, Recall, etc.

### Discussion  
* Compare performance results to other studies
* Outcome selection
    - data most correlated with CMIS_MATCH (interesting)
* Predictor selection
    - no performance change between grouped, ungrouped billing features (interesting bc landlord likely pays some)
    - selected those that are likely to be widely available across US
* More appropriate methods for dealing with time component - LSTM (other study)
    - P/N cases did separate more at certain points in time (not immediately before experiencing homelessness)

### Conclusion  
* Utility billing data contains important info in predicting homelessness
* Combining data from multiple sources (medical, DOJ, financial) seems best approach
* National homelessness data collection + prediction program
* Future research: combining utility data with other useful data

---
## General
### Which metrics and at what threshold to report?
* Default is 0.5, but might not be most favorable
* Report all metrics listed in all other relevant studies to make comparable?
### How to compare to other studies? 
Study 1 below uses cutoff threshold of 0.5 (the default threshold) to record metrics, but our model may compare more favorably if we use a different threshold.

---
## Changes By Section
### Abstract
### Introduction
### Methods
* Clarify data preprocessing regarding time - only looking at predictors before homelessness is experienced
### Results
* Report area under ROC, balanced accuracy
### Discussion
### Conclusion
* Add conclusion

---
## Studies to Add
* Study 1: [A classification model of homelessness using integrated administrative data: Implications for targeting interventions to improve the housing status, health and well-being of a highly vulnerable population](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237905)
* Study 2: [An Open-Source Interpretable Machine Learning Approach to Prediction of Chronic Homelessness](https://towardsdatascience.com/an-open-source-interpretable-machine-learning-approach-to-prediction-of-chronic-homelessness-8215707aa572)
* [Original Paper](https://arxiv.org/abs/2009.09072v1)

---
# Journal Prospects
## PLOS Global Public Health
[Getting Started](https://journals.plos.org/plosone/s/getting-started)  
### Fee
[\$2,100](https://plos.org/publish/fees/)

---
## Frontiers in Applied Mathematics and Statistics
### Fee
\$1,150  
### Restrictions  
Article Type: Original Research (A-type article)
* Abstract length: 350 words
* Maximum word count of 12,000
* No more than 15 Figures/Tables  
### Format  
1) Abstract
2) Introduction
3) Materials and Methods
4) Results
5) Discussion