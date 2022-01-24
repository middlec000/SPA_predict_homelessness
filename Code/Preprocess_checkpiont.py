import pandas as pd
import numpy as np
import pickle
import time
from custom_methods import preprocessing, calc_time
from helper_methods import accumulate

filename = 'processed0.pickle'
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

rows = len(df)
accts = df.SPA_ACCT_ID.nunique()
premises = df.SPA_PREM_ID.nunique()
print(f'Length: {rows}')
print(f'Accounts: {accts}')
print(f'Premises: {premises}')

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
print(f'Earliest Month: {df.MONTH.min()}')
print(f'Latest Month: {df.MONTH.max()}')
df.MONTH = df.MONTH.apply(lambda x: preprocessing.date_map(date=x, relative_to=201512, format='yyyymm'))
print(f'Earliest Month: {df.MONTH.min()}')
print(f'Latest Month: {df.MONTH.max()}')

# Prepare for Matching
df = df.drop_duplicates()

df = df[~df.TOTAL_CUR_BALANCE.isna()]

# Change NA for the following attributes to 0
# Assume no data means there have been 0 occurrances of each of these
df['BREAK_ARRANGEMENT'] = df['BREAK_ARRANGEMENT'].replace(to_replace=np.nan, value=0.0)
df['PAST_DUE'] = df['PAST_DUE'].replace(to_replace=np.nan, value=0.0)

# Just choose last of duplicates
print(f'Original length: {rows}')
print(f'Original accounts: {accts}')
print()
df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()
print(f'New length: {len(df)}')
print(f'New accounts: {df.SPA_ACCT_ID.nunique()}')
print(f'\n{rows-len(df)} Rows lost')
print(f'{accts-df.SPA_ACCT_ID.nunique()} Accounts lost\n')

print(f"Duplicates? \n{df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()}")

# Attach Service Agreements Data
sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').\
    rename({'spa_prem_id':'SPA_PREM_ID', 'spa_acct_id':'SPA_ACCT_ID', 'spa_per_id':'SPA_PER_ID', 'homelessMatch':'CMIS_MATCH', 'EnrollDate':'ENROLL_DATE', 'apartment':'APARTMENT'}, axis=1)
sa_rows = len(sa)
sa_ppl = sa.SPA_PER_ID.nunique()
sa_accts = sa.SPA_ACCT_ID.nunique()
sa_pos_ppl = sa[sa.CMIS_MATCH == True].SPA_PER_ID.nunique()

print(f'Rows: {sa_rows}')
print(f'Accounts: {sa_accts}')
print(f'People: {sa_ppl}')
print(f'Positive Cases: {sa_pos_ppl}\n')

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

# Any null enroll dates for cmis_match? No - good
print(f'Null Enroll Dates for P Cases: {sa[sa.CMIS_MATCH]["ENROLL_DATE"].isnull().sum()}')

# Retain only columns we want to add to billing - note: all CMIS_MATCHes have ENROLL_DATEs
sa.drop(['spa_sa_id', 'START_DT', 'END_DT', 'SA_TYPE_DESCR', 'Class'], axis=1, inplace=True)

# Create list of accounts that have a cotenant
cotenant_accounts = sa[sa['ACCT_REL_TYPE_CD'] == 'COTENANT']['SPA_ACCT_ID'].values
# Only keep info regarding the 'MAIN' account holder
sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN'].drop('ACCT_REL_TYPE_CD', axis=1)
# Add boolean column for cotenant
sa['HAS_COTENANT'] = sa['SPA_ACCT_ID'].isin(cotenant_accounts).astype('bool')
del cotenant_accounts
sa.drop_duplicates(inplace=True)

# Group Enroll Dates into list
enroll_dates = sa[~sa["ENROLL_DATE"].isnull()].groupby(["SPA_ACCT_ID", "SPA_PREM_ID"])["ENROLL_DATE"].unique()
sa = sa.set_index(['SPA_ACCT_ID','SPA_PREM_ID']).sort_index()
sa.update(enroll_dates)
del enroll_dates
sa["ENROLL_DATE"] = sa["ENROLL_DATE"].apply(lambda x: tuple([]) if np.isnan(x).all() else tuple(x))

# If any CMIS_MATCH for person, then CMIS_MATCH for all instances of person
sa.update(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID'])["CMIS_MATCH"].any())
sa.drop_duplicates(inplace=True)
sa.reset_index(inplace=True)

print(f'Positive Cases: {sa[sa.CMIS_MATCH].SPA_PER_ID.nunique()}')

# Check Matching
print('\nGrouping:')
print(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID']).size().value_counts())

sa = sa.drop(['APARTMENT', 'HAS_COTENANT'], axis=1)

sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='inner')

del sa

print(f'Number of Positives: {df[df.CMIS_MATCH].SPA_PER_ID.nunique()}')

"""
## Fix Duplicate Identifiers
Problem  
* Multiple instances of the same ('SPA_PER_ID', 'SPA_PREM_ID', 'MONTH') combination  

Cause  
* Different accounts at same time - person switched accounts for some reason  

Solution  
* Pick last
"""
rows1 = len(df)
print(f'Original length: {rows1}')
print()
df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()
print(f'New length: {len(df)}')
print(f'\n{rows1-len(df)} Rows lost')

print(f"Duplicates? \n{df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()}")

# Drop data that is post-ENROLL_DATE
df['min_enroll'] = df['ENROLL_DATE'].apply(lambda x: np.nan if x is np.nan else min(x))
df['drop_me'] = (df['MONTH'] > df['min_enroll'])

# Create ID
df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')

df = df.drop(['drop_me', 'min_enroll', 'ENROLL_DATE'], axis=1)

# Create additional features
# Determine cumulative number of places a person has paid bills at so far
df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')

# Determine cumulative number of people a premesis has seen so far
df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')

df = df.drop(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH'], axis=1)

# Check nulls
print(f"Nulls: {df.isnull().sum().sum()}")

# Save
outfile = open(datapath+filename, 'wb')
pickle.dump(df, outfile)
outfile.close()

# Check retained
retained_rows = len(df)
retained_accts = df.SPA_ACCT_ID.nunique()
retained_pos_cases = df[df.CMIS_MATCH].SPA_PER_ID.nunique()

print(f'Retained {retained_rows} = {100*retained_rows/rows}% of rows.')
print(f'Retained {retained_accts} = {100*retained_accts/accts}% of accounts.')
print(f'Retained {retained_pos_cases} = {100*retained_pos_cases/sa_pos_ppl}% of positive cases.')
print(f'People: {df.SPA_PER_ID.nunique()}')
print(f'Negative Cases: {df[~df.CMIS_MATCH].SPA_PER_ID.nunique()}')

print(calc_time.calc_time_from_sec(time.time()-startTime))