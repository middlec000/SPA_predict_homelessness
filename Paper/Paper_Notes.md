# TODO
## Make data available
Urbanova or City of Spokane system?  
Determine which release(s) used. Combine billing data files.

## Conform to PLOS paper standards
See below

## Ask for Feedback From:
* Kim Boynton
* David Lewis

## Publish Preprint
arXiv -> Statistics -> Applications  
(wait until manuscript is ready)

---
# Feedback from Dr. Oster

---
# Questions/Notes for Dr. Oster
Emailed with Dr. Dan Li and she said standard model reporting is not done, just that studies typically report accuracy, precision (PPV), recall (TPR), F1-score, and the ROC Curve.

Only used PIT counts for general homelessness numbers, not for prediction. Used CMIS data for prediction (not homelessness numbers). CMIS data is better for us because they track people (name, last address, etc.), but is likely less complete than the PIT counts (collect counts from shelters and walk around and count unsheltered).  

Did not directly look at variance of predicted risk level before taking maximum for each person, but I did try to use mean predicted risk instead of maximum - much worse performance. Perhaps taking mean of 3 highest predicted risks (or similar) would be a good approach. I'd rather not investigate this now, though.  

Reason for not using Cox Regression. I think you are right about our Logistic Regression + taking maximum over time and Cox Time Varying Covariates being essentially equivalent. I'm hesitant to say they are mathematically equivalent, but they do perform the same functionality. I updated this in the Discussion -> Dependence on Time subsection.  

In response to Discussion -> Data Limitations: "I wonder if a high proportion of those identified had both City and Avista bills (was there bias?)."   
I did not look at this. Theoretically if the City and Avista bill amounts owed are kept separate, the model can get more information out of them, but I compared model performance and it is essentially the same. The nice thing about the aggregated version is that it is more generic so places where utilities are combined could still use this same model.  

I think we should leave model performance in the Abstract - other papers do this.  

Publish preprint?  

Plos says to remove figures and keep them separate from .tex file - remove non-image figures such as Fig 2. (Preprocessing Diagram) and Fig 3. ((Binning Threshold Examples)?

---
# PLOS ONE
Fee = [\$1,805](https://plos.org/publish/fees/)  

[Getting Started](https://journals.plos.org/plosone/s/getting-started)  

## Make sure your manuscript is a good fit for PLOS ONE
### [Criteria for Publication](https://journals.plos.org/plosone/s/criteria-for-publication)
1. The study presents the results of original research - done
2. Results reported have not been published elsewhere - done
3. Experiments, statistics, and other analyses are performed to a high technical standard and are described in sufficient detail - done
4. Conclusions are presented in an appropriate fashion and are supported by the data - done
5. The article is presented in an intelligible fashion and is written in standard English - done
6. The research meets all applicable standards for the ethics of experimentation and research integrity - done
7. The article adheres to appropriate reporting guidelines and community standards for data availability
   1. PLOS journals require authors to make all data necessary to replicate their study’s findings publicly available without restriction at the time of publication. When specific legal or ethical restrictions prohibit public sharing of a data set, authors must indicate how others may obtain access to the data.

## [Identify potential editors](https://journals.plos.org/plosone/static/editorial-board)
Public health related  
Benn Sartorius was editor of Byrne et al.'s "A classification model of homelessness using integrated administrative data: Implications for targeting interventions to improve the housing status, health and well-being of a highly vulnerable population" published by PLOS ONE

## Read journal policies
### [Data Availability](https://journals.plos.org/plosone/s/data-availability)
### [Materials, Software and Code Sharing](https://journals.plos.org/plosone/s/materials-software-and-code-sharing) - Done
Linked directly from the manuscript file - done

## Prepare your data
Need to do this

## Confirm the author list
* Full names, including initials if used
* Affiliations
* Email address
* Any potential competing interests
* Funding information
* Author contributions
* Order of authors

### Title, Author List, Affiliations
### Body Formatting
* Headings should be written in sentence case - done
* All text in one .tex file - done
* No graphics in manuscript submission
  * Figures: uploaded separately
* Tables: no graphics or colored text (cell background color/shading OK) - done

## Check that your study follows appropriate reporting guidelines - Done
Our study does not fall into any of the specific reporting guidelines listed.

## Read the license agreement
## Prepare funding and competing statements
* Disclosure of funding sources
* Competing interests (None) - Done

## Consider posting a preprint - Done
Not doing one

## Access the submission system

# PLOS Global Public Health
Fee = [\$2,100](https://plos.org/publish/fees/)  

[Submission Guidelines](https://journals.plos.org/globalpublichealth/s/submission-guidelines)  
Essentially the same as for PLOS ONE (above)  
[Figure Guidelines](https://journals.plos.org/globalpublichealth/s/figures)

# Data Sharing
## Meetings
03/22/22
Kim Boynton
What is Avista's stance on data sharing? Urbanova or City of Spokane?
Avista yes through Urbanova with data sharing agreement
  each entity would have to join data sharing agreement
  Meeting w/Mason on Thursday 

SPA 2
Urbanova -> Google Cloud
  Some cloud computing

Quality issues with collections data - need to improve

Can call Kim