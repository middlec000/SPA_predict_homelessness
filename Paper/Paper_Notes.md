# Changes to Paper
## New Outline
### Introduction  
* State of homelessness in USA  
* Precedent Research
  * Other studies focus on predicting shelter reentry or high cost from homeless population
  * Richer data sources when focusing on those already in service system:
      - Medical (emergency services)
      - Mental health
      - Financial assistance usage
      - Department of Justice
* Predicting homelessness on large scale (state, national) preferable
    - Lose tracking of fewer people since they all stay in same larger region
    - Additional information sources when people move
    - Additional cost savings of preventing first-time homelessness
  * Want financial info
      - Taxes, salary, etc.
      - We use utility billing history as proxy
* Goals of this Study
    - Determine usefulness of ubiquitous utility customer billing data
    - Create model that produces few false-negatives

### Methods  
* Data
    - Sources
    - Description
    - Preparation
    - Feature engineering and selection
      - Outcome Measure
      - Predictor Measures
        - selected those that are likely to be widely available across US
      - Engineered Features
    - Limitations
      - Turned out data coverage still lacking
      - 
* Model
    - Logistic on person-place-month
    - Maximum risk for person
* Evaluation
    - K-Folds
    - ROC curve
    - Other Metrics

### Results  
* ROC curve
* Other metrics

### Discussion  
* Compare performance results to other studies
* Outcome selection
    - data most correlated with CMIS_MATCH, not others tried (interesting)
* Predictor selection
    - no performance change between grouped, ungrouped billing features (interesting bc landlord likely pays some)
    - Selected those that are likely to be widely available across US
    - Difficulty using geographical predictors
      - Too many levels
* More appropriate methods for dealing with time component - LSTM (other study)
    - P/N cases did separate more at certain points in time (not immediately before experiencing homelessness)
    - Socioeconomic "pathology" of homelessness over time

### Conclusion  
* Utility billing data contains important info in predicting homelessness
* Combining data from multiple sources (medical, DOJ, financial) likely best approach
* Standardized reporting
* National homelessness data collection + prediction program
* Future research: combining utility data with other useful data, study cost savings of preventing first-time homelessness

### Supporting Documents
* Model parameters
  * Model assumptions not met so cannot be interpreted meaningfully, only heuristically

---
## General
### Organization
* Move tables and figures to another section as in example

---
## Changes By Section
### Abstract
### Introduction
### Methods
* Clarify data preprocessing regarding time - only looking at predictors BEFORE homelessness is experienced
### Results
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