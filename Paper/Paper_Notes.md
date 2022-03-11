# TODO
## Remove figures before publication
## Add info about how to access data
Publish on GitHub with code?

## Conform to PLOS paper standards
See below

## Ask for Feedback From:
* Kim Boynton
* David Lewis

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

### [Title, Author List, Affiliations](https://storage.googleapis.com/plos-published-prod/ba62/PLOSOne_formatting_sample_title_authors_affiliations.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wombat-sa%40plos-prod.iam.gserviceaccount.com%2F20220311%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220311T154107Z&X-Goog-Expires=86400&X-Goog-SignedHeaders=host&X-Goog-Signature=3fc685b2ddd04242ec1bb83016e99ed1ea32c7c533a5094263804817bcf765a0874c355bc34682ac2fef17767a3a4247a571b2c7588dc843b2446fa484412fa63ad6632325d56ea14c0d197ebb8c179cc48a22193f9600830146bc9087556283abbde96e3c80d4cba394cbcc26fa5ae6e2f3e8ea6da31f51eeaaf0154db6ab8f0c2d4c8f30046c0d178d97710bebe3d134131d9aac7b4d650316409318b14d2bc94d5dc9f38a139a02e01120549408ed788f5662bd55c63662117fd7ca833fa104cbe542bbce96de2a7c766696317e3fac0aae0fb0cfab68d09434059629a75a5389b338ffb93b4f6659fe83a625a4cc3095aa45bdd1b6f5fade911cb61a75eb)
### [Body Formatting](https://storage.googleapis.com/plos-published-prod/wjVg/PLOSOne_formatting_sample_main_body.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=wombat-sa%40plos-prod.iam.gserviceaccount.com%2F20220311%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220311T153202Z&X-Goog-Expires=86400&X-Goog-SignedHeaders=host&X-Goog-Signature=c9b2a6d6b43ac607acc8abcba656b2616eb3a5dcc965fabc7c1996a0d65959ab8e58045263bea631b26c045dc9728cf5e1b9d538a01f2f283480997bbe18a7fe2e9ad1b2125ad4fe189b22cbeb73298fb3338f38daaf0e9d12597c7f6bf28fefa79021873570665b87b8452e6f946be74fc981c21dbb4a721d9006ba27874074b65dc87c3bb66a27779025b8c69c3f5b4cb8d9b756f346a9916e4f6cb26d972422bcd4c6f1244fa1d5f4262fe17de5957a96e61f300e51cd3bf1a62fa6467f25bddc541f90b2675bfa87873715aeec4bb30162603bbfdae15bf1cd95b6f9f7f156411bb38e254069eca4a6026d75555ddceea6221199e04b9001f6af64d4c7ea)
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

## Consider posting a preprint
Do this?

## Access the submission system