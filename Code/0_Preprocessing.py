import pandas as pd
import numpy as np
import pickle
import time
from Code.log_helper_methods import generate_log
from custom_methods import preprocessing, calc_time
from log_helper_methods import accumulate

filename = 'processed_all_ids.pickle'
datapath = '../Data/'

# start timer
startTime = time.time()

# Load Billing Data
# This just grabs all the seperate billing data files
fileyears = ['2015', '2016', '2017', '2018', '2019', '2020']
path = 'SpaData_'
df = pd.read_csv(datapath+path+fileyears[0]+'_Anon.csv')
for fileyear in fileyears[1:]:
    df = df.append(pd.read_csv(datapath+path+fileyear+'_Anon.csv'))

original_billing_stats = {
    'rows': len(df),
    'accounts': df.SPA_ACCT_ID.nunique(),
    'premisses': df.SPA_PREM_ID.nunique()
}
print(f'Original Billing Stats\n{original_billing_stats}')

df = df.rename({'ARREARSMONTH':'MONTH'}, axis=1)

# Create additional attribute
df['TOTAL_CUR_BALANCE'] = df['RES_EL_CUR_BAL_AMT'] + df['RES_GAS_CUR_BAL_AMT'] + df['CITY_TOT_DUE']

keep = [
    'SPA_PREM_ID',
    'SPA_ACCT_ID',
    'MONTH',
    'TOTAL_CUR_BALANCE',
    'BREAK_ARRANGEMENT',
    'PAST_DUE'
]

# Only keep desired predictors
df = df[keep]

# Reformat Dates
df.MONTH = df.MONTH.apply(lambda x: preprocessing.date_map(date=x, relative_to=201512, format='yyyymm'))

# Prepare for Matching
df = df.drop_duplicates()

df = df[~df.TOTAL_CUR_BALANCE.isna()]

# Change NA for the following attributes to 0
# Assume no data means there have been 0 occurrances of each of these
df['BREAK_ARRANGEMENT'] = df['BREAK_ARRANGEMENT'].replace(to_replace=np.nan, value=0.0)
df['PAST_DUE'] = df['PAST_DUE'].replace(to_replace=np.nan, value=0.0)

# Just choose last of duplicates
df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

# Attach Service Agreements Data
sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').\
    rename({'spa_prem_id':'SPA_PREM_ID', 'spa_acct_id':'SPA_ACCT_ID', 'spa_per_id':'SPA_PER_ID', 'homelessMatch':'CMIS_MATCH', 'EnrollDate':'ENROLL_DATE', 'apartment':'APARTMENT'}, axis=1)

original_sa_stats = {
    'rows': len(sa),
    'accounts': sa.SPA_ACCT_ID.nunique(),
    'premisses': sa.SPA_PREM_ID.nunique(),
    'people': sa.SPA_PER_ID.nunique(),
    'pos_people': sa[sa.CMIS_MATCH == True].SPA_PER_ID.nunique(),
    'neg_people': sa.SPA_PER_ID.nunique() - sa[sa.CMIS_MATCH == True].SPA_PER_ID.nunique()
}
print(f'Original Service Agreement Stats\n{original_sa_stats}')

"""
Problems:
* Some accounts have multiple people associated with them at a time, some only have one
* Some people are associated with multiple accounts (sometimes at different 'ACCT_REL_TYPE_CD')  

Solution:  
* Only retain the main account holder for each account
"""

# Convert Dates to months since December, 2015
sa.ENROLL_DATE = sa.ENROLL_DATE.apply(lambda x: preprocessing.date_map(date=x, relative_to='2015-01-01', format='yyyy-mm-dd'))

# Replace NaN with False in CMIS_MATCH
sa.CMIS_MATCH = sa.CMIS_MATCH.replace(to_replace=np.nan, value=False).astype('bool')

# Retain only columns we want to add to billing - note: all CMIS_MATCHes have ENROLL_DATEs
sa.drop(['spa_sa_id', 'START_DT', 'END_DT', 'SA_TYPE_DESCR', 'Class'], axis=1, inplace=True)

# Only keep info regarding the 'MAIN' account holder
# sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN'].drop('ACCT_REL_TYPE_CD', axis=1)

# Take min ENROLL_DATE
#enroll_dates = sa[~sa["ENROLL_DATE"].isnull()].groupby(["SPA_ACCT_ID", "SPA_PREM_ID"])["ENROLL_DATE"].unique()
enroll_dates = sa[~sa["ENROLL_DATE"].isnull()].groupby(["SPA_ACCT_ID", "SPA_PREM_ID"])["ENROLL_DATE"].min()
sa = sa.set_index(['SPA_ACCT_ID','SPA_PREM_ID'])
sa.update(enroll_dates)
del enroll_dates
# sa["ENROLL_DATE"] = sa["ENROLL_DATE"].apply(lambda x: tuple([]) if np.isnan(x).all() else tuple(x))

# If any CMIS_MATCH for person, then CMIS_MATCH for all instances of person
sa.update(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID'])["CMIS_MATCH"].any())
sa.drop_duplicates(inplace=True)
sa.reset_index(inplace=True)

sa = sa.drop(['APARTMENT', 'ACCT_REL_TYPE_CD'], axis=1)

sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='inner')

del sa

"""
## Fix Duplicate Identifiers
Problem  
* Multiple instances of the same ('SPA_PER_ID', 'SPA_PREM_ID', 'MONTH') combination  

Cause  
* Different accounts at same time - person switched accounts for some reason  

Solution  
* Pick last
"""
# Coerce id columns to ints
to_ints = ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH', 'SPA_ACCT_ID']
for col in to_ints:
    df[col] = df[col].astype('int')

df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

# Drop data that is post-ENROLL_DATE
# df['min_enroll'] = df['ENROLL_DATE'].apply(lambda x: np.nan if x is np.nan else min(x))
df['drop_me'] = (df['MONTH'] > df['ENROLL_DATE'])
df = df[~df['drop_me']]

# Create combined ID
df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')

df = df.drop(['drop_me', 'ENROLL_DATE'], axis=1)

# Create additional features
# Determine cumulative number of places a person has paid bills at so far
df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')

# Determine cumulative number of people a premesis has seen so far
df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')

# df = df.drop(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH'], axis=1)

# Check nulls
print(f"\nNulls: \n{df.isnull().sum().sum()}")

# Check Grouping
print(f"\nDuplicates? \n{df.groupby(['PER-PREM-MONTH_ID']).size().value_counts()}")

# Check Data Types
# print(f'Data Types: \n{df.dtypes}')

# Save
output = {
    'Data': df,
    'Features': df.columns.to_list(),
    'Data_Retention_Stats': generate_log(
        original_billing=original_billing_stats,
        original_sa=original_billing_stats,
        final_df=df
        )
}

outfile = open(datapath+filename, 'wb')
pickle.dump(output, outfile)
outfile.close()

# Print Time
print(calc_time.calc_time_from_sec(time.time()-startTime))