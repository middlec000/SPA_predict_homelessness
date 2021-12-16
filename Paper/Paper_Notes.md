# Changes to Paper
## New Outline
### Introduction  
* State of homelessness in USA  
* Predicting homelessness on large scale (state, national) preferable
    - Lose tracking of fewer people since they all stay in same larger region
    - Additional information sources when people move
* Goals
    - Determine usefulness of ubiquitous utility customer billing data
    - Create model that produces few false-negatives
* Precedent Research
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
    - Description
    - Preparation
    - Feature engineering and selection
    - Limitations
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
### Organization
* Move tables and figures to another section as in example

---
## Changes By Section
### Abstract
### Introduction
### Methods
* Clarify data preprocessing regarding time - only looking at predictors before homelessness is experienced
* Update variables used
### Results
* Report area under ROC, balanced accuracy
* Report at threshold of 0.5
* Report at 90% TPR
* Report mean model coefficients
### Discussion
VanBerlo (2020) ML model:  
tpr: 92.1  = tp / p = tp / (tp + fn)  
ppv: 65.1  = tp / (tp + fp)  
F1-score: 76.3  
AUC: 97.6  
Accuracy: 97.1  = (tp + tn) / total  
calculated:  
fpr: fp / n = fp / (fn + fp)  
  
Byrne (2020) log model:  
AUC: 94.0  
balanced accuracy: 86.4  
tpr: 77.8  
tnr: 95.1  
ppv: 11.7  
npv: 99.8  
calculated:
fpr = 1 - tnr = 1 - 0.951 = 0.049  

Shinn (2013) screening model (points based):  
screening model:  
tpr: 91.9, 74.7  
fpr: 65.7, 36.6  

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
### Manuscript Specifications
* All text in one .tex file
* No graphics in manuscript submission
* Figures: uploaded separately
* Figures: Use Fig instead of Figure
* Tables: NO spacing/line breaks within cells to alter layout or alignment
* Tables: do not nest tabular environments (no tabular environments within tabular environments)
* Tables: no graphics or colored text (cell background color/shading OK)
* Tables: For tables that exceed the width of the text column, use the adjustwidth environment
* Math: Do not include text that is not math in the math environment. For example, CO2 should be written as CO\textsubscript{2} instead of CO\$_2\$.
* Math: When adding superscript or subscripts outside of brackets/braces, please group using {}.  For example, change "[U(D,E,\gamma)]^2" to "{[U(D,E,\gamma)]}^2". 

---
## Frontiers in Applied Mathematics and Statistics
### Fee
\$1,150  
### Manuscript Specifications
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

# Editing Tracking
## Figure Labels
fig:Homeless_US_Spokane  
fig:corrRankByYear  
fig:corr_years  
fig:PN_mo_away_on_TOTAL_60_DAYS_AMT  
fig:cat_theil  
fig:ROC  
## Table Labels
tbl:assocWithOutcome15  
tbl:varsUsed  
tbl:evalTerminology  
tbl:performance  
## Other Labels
fig1
table1