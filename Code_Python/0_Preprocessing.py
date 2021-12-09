# Setup
import pandas as pd
import numpy as np
import pickle
import time
from custom_methods import preprocessing, calc_time

datapath = '../Data/'

# start timer
startTime = time.time()


# Billing Data
# Load Billing Data
# This just grabs all the separate billing data files
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
df = df.drop(['MONTHID', 'COVID_REMINDER'], axis=1)

print(f'Earliest Month: {df.MONTH.min()}')
print(f'Latest Month: {df.MONTH.max()}')

# Use December, 2015 as month 0 - this is the earliest month in the billing data

df.MONTH = df.MONTH.apply(lambda x: preprocessing.date_map(date=x, relative_to=201512, format='yyyymm'))
print(f'Earliest Month: {df.MONTH.min()}')
print(f'Latest Month: {df.MONTH.max()}')


# ## Prepare for Matching
# Want to match on unique combinations of ('SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH')

df = df.drop_duplicates()

print(df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts())
print(df.isnull().sum())

df = df[~df.CITY_TOT_DUE.isna()]

# Change NA for the following attributes to 0
# Assume no data means there have been 0 occurrences of each of these
set_to_zero = [
    'BREAK_ARRANGEMENT',
    'BREAK_PAY_PLAN',
    'CALL_OUT',
    'CALL_OUT_MANUAL',
    'DUE_DATE',
    'FINAL_NOTICE',
    'PAST_DUE',
    'SEVERANCE_ELECTRIC',
    'SEVERANCE_GAS',
]

for col in set_to_zero:
    df[col] = df[col].replace(to_replace=np.nan, value=0)

# Just choose last of duplicates
print(f'Original length: {rows}')
print(f'Original accounts: {accts}')
print()
df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()
print(f'New length: {len(df)}')
print(f'New accounts: {df.SPA_ACCT_ID.nunique()}')
print(f'\n{rows-len(df)} Rows lost')
print(f'{accts-df.SPA_ACCT_ID.nunique()} Accounts lost\n')

df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()

# # Service Agreements
sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').    rename({'spa_prem_id':'SPA_PREM_ID', 'spa_acct_id':'SPA_ACCT_ID', 'spa_per_id':'SPA_PER_ID', 'homelessMatch':'CMIS_MATCH', 'EnrollDate':'ENROLL_DATE', 'apartment':'APARTMENT'}, axis=1)
sa_rows = len(sa)
sa_ppl = sa.SPA_PER_ID.nunique()
sa_accts = sa.SPA_ACCT_ID.nunique()
sa_pos_ppl = sa[sa.CMIS_MATCH == True].SPA_PER_ID.nunique()

print(f'Rows: {sa_rows}')
print(f'Accounts: {sa_accts}')
print(f'People: {sa_ppl}')
print(f'Positive Cases: {sa_pos_ppl}\n')
sa.head()


# ## Transform
# Problems:
# * Some accounts have multiple people associated with them at a time, some only have one
# * Some people are associated with multiple accounts (sometimes at different 'ACCT_REL_TYPE_CD')  
# 
# Solution:  
# * Only retain the main account holder for each account
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


# ## Join to df


sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='left')

del sa

df.isnull().sum()


# ## Only Keep Known SPA_PER_ID's

# In[13]:


df = df[~df.SPA_PER_ID.isna()]
df.isnull().sum()


# In[14]:


df[df.CMIS_MATCH].SPA_PER_ID.nunique()


# # Check Identifiers

# In[15]:


df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()


# In[16]:


df[df.duplicated(subset=['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH'], keep=False)]


# ## Fix Duplicate Identifiers
# Problem  
# * Multiple instances of the same ('SPA_PER_ID', 'SPA_PREM_ID', 'MONTH') combination  
# 
# Cause  
# * Different accounts at same time - person switched accounts for some reason  
# 
# Solution  
# * Pick last

# In[17]:


rows1 = len(df)
print(f'Original length: {rows1}')
print()
df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()
print(f'New length: {len(df)}')
print(f'\n{rows1-len(df)} Rows lost')

df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).size().value_counts()


# # Geo - Avista

# In[18]:


geo = pd.read_csv(datapath+'GeoData_Anon.csv').rename({'spa_prem_id':'SPA_PREM_ID'}, axis=1)

geo = geo.drop(["TRACT_GEOID", "BLOCKGROUP_GEOID_Data"], axis=1).drop_duplicates()

# NOTE: BLOCKGROUP_GEOID and BLOCKGROUP_GEOID_Data contain the same blockgroup number
print(f'Total Records: {len(geo)}')
print(f'Total Premises: {geo.SPA_PREM_ID.nunique()}')
print(f"Contains NaN's: {geo.isnull().any().any()}")
geo.head()


# In[19]:


df = df.join(geo.set_index('SPA_PREM_ID'), on=['SPA_PREM_ID'], how='left')
del geo

df.isnull().sum()


# ## Multi-Family Dwellings

# In[20]:


dwellings = pd.read_csv(datapath+'MultiFamilyDwellingIDs_Anon.csv').rename({'spa_prem_id':'SPA_PREM_ID', 'multi_dwell_id':'MULTI_DWELL_ID'}, axis=1)
df = df.join(dwellings.set_index('SPA_PREM_ID'), on=['SPA_PREM_ID'], how='left')
del dwellings


# # Geo Data - Census
# Using data from 2015

# In[21]:


sub_datapath = datapath+'CensusData/'
match_col = 'BLOCKGROUP_GEOID'


# ## Aggregate Income
# US Census Table: B19025

# In[22]:


agg_income = pd.read_csv(sub_datapath+'AggIncome/ACSDT5Y2015.B19025_data_with_overlays_2021-04-18T191340.csv')

agg_income.drop(0, axis=0, inplace=True)
newcol = "AGG_INCOME_GEO"
agg_income.rename({"B19025_001E":newcol}, axis=1, inplace=True)

agg_income[match_col] = agg_income["GEO_ID"].map(preprocessing.geoid_map).astype('int64')
agg_income.set_index(match_col, inplace=True)

df = df.join(agg_income[newcol], how='left', on=match_col)

del agg_income


# ## Earnings
# US Census Table: B19051

# In[23]:


earnings = pd.read_csv(sub_datapath+'Earnings/ACSDT5Y2015.B19051_data_with_overlays_2021-04-12T234426.csv')

earnings = earnings.drop(0, axis=0)
newcol = "NO_EARNINGS_GEO"
earnings[newcol] = earnings["B19051_003E"].astype('float') / earnings["B19051_001E"].astype('float')
earnings[match_col] = earnings["GEO_ID"].map(preprocessing.geoid_map).astype("int64")
earnings = earnings.set_index(match_col)

df = df.join(earnings[newcol], how='left', on=match_col)
del earnings


# ## Poverty
# US Census Table B17021

# In[24]:


poverty = pd.read_csv(sub_datapath+'Poverty/ACSDT5Y2015.B17021_data_with_overlays_2021-04-12T234708.csv')

poverty.drop(0, axis=0, inplace=True)
newcol = "BELOW_POVERTY_LVL_GEO"
poverty[newcol] = poverty["B17021_002E"].astype('float') / poverty["B17021_001E"].astype('float')
poverty[match_col] = poverty["GEO_ID"].map(preprocessing.geoid_map).astype('int64')
poverty = poverty.set_index(match_col)

df = df.join(poverty[newcol], how='left', on=match_col)
del poverty


# ## Food Stamps / SNAP
# US Census Table: B22010

# In[25]:


snap = pd.read_csv(sub_datapath+'FoodStamps/ACSDT5Y2015.B22010_data_with_overlays_2021-04-18T182516.csv')
snap.drop(0, axis=0, inplace=True)
newcol = "SNAP_GEO"
snap[newcol] = snap["B22010_002E"].astype('float') / snap["B22010_001E"].astype('float')
snap[match_col] = snap["GEO_ID"].map(preprocessing.geoid_map).astype('int64')
snap = snap.set_index(match_col)

df= df.join(snap[newcol], how='left', on=match_col)
del snap


# ## Education Attainment
# US Census Table: B15003

# In[26]:


edu = pd.read_csv(sub_datapath+'Education/ACSDT5Y2015.B15003_data_with_overlays_2021-04-18T184101.csv')
edu.columns = edu.iloc[0]
edu = edu.drop(0, axis=0)
# Get rid of margin of error columns
for col in edu.columns:
    if col == "Margin of Error":
        edu = edu.drop(col, axis=1)

newcol = "ABOVE_GRD7_GEO"
# Sum of people above Grade 7 / total
edu[newcol] = edu.iloc[:,13:].sum(axis=1) / edu.iloc[:,3:].sum(axis=1)

edu[match_col] = edu["id"].map(preprocessing.geoid_map).astype('int64')
edu = edu.set_index(match_col)

df = df.join(edu[newcol], how='left', on=match_col)
del edu


# ## Public Assistance¶
# US Census Table: B19057

# In[27]:


assist = pd.read_csv(sub_datapath+'PublicAssistance/ACSDT5Y2015.B19057_data_with_overlays_2021-04-18T190814.csv')
assist.drop(0, axis=0, inplace=True)
newcol = "PUBLIC_ASSIST_GEO"

assist[newcol] = assist["B19057_002E"].astype('float') / assist["B19057_001E"].astype('float')
assist[match_col] = assist["GEO_ID"].map(preprocessing.geoid_map).astype('int64')
assist = assist.set_index(match_col)

df = df.join(assist[newcol], how='left', on=match_col)
del assist


# # Check Data

# ## Check Nulls

# In[28]:


df.isnull().sum()


# In[29]:


# Drop all NA Geo info
df = df[~df.POSTAL.isna()]

df.isnull().sum()


# ## Check Data Types

# In[30]:


df.dtypes


# In[31]:


# If premesis not a multi-unit dwelling, set ID to -1
df.MULTI_DWELL_ID = df.MULTI_DWELL_ID.replace(to_replace=np.nan, value=-1)

to_ints = [
    'BREAK_ARRANGEMENT',
    'BREAK_PAY_PLAN',
    'CALL_OUT',
    'CALL_OUT_MANUAL',
    'DUE_DATE',
    'FINAL_NOTICE',
    'PAST_DUE',
    'SEVERANCE_ELECTRIC',
    'SEVERANCE_GAS',
    'SPA_PREM_ID',
    'SPA_ACCT_ID',
    'SPA_PER_ID',
    'BLOCKGROUP_GEOID',
    'POSTAL',
    'MULTI_DWELL_ID',
]
for col in to_ints:
    df[col] = df[col].astype('int')

to_bools = [
    'CMIS_MATCH',
    'APARTMENT',
    'HAS_COTENANT',
]
for col in to_bools:
    df[col] = df[col].astype('bool')

to_floats = [
    'AGG_INCOME_GEO',
]
for col in to_floats:
    df[col] = df[col].astype('float')

df.dtypes


# # Create Additional Attributes

# ## ID Combination

# In[32]:


df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')


# ## Billing Aggregation

# In[33]:


# Avista, City level
df['AVISTA_CUR120_DAYS'] = df['RES_EL_CUR120_DAYS'] + df['RES_GAS_CUR120_DAYS']
df['AVISTA_OVER_120_DAYS'] = df['RES_EL_OVER_120_DAYS'] + df['RES_GAS_OVER_120_DAYS']
df['AVISTA_CUR22_DAYS'] = df['RES_EL_CUR22_DAYS'] + df['RES_GAS_CUR22_DAYS']
df['AVISTA_CUR30_DAYS'] = df['RES_EL_CUR30_DAYS'] + df['RES_GAS_CUR30_DAYS']
df['AVISTA_CUR60_DAYS'] = df['RES_EL_CUR60_DAYS'] + df['RES_GAS_CUR60_DAYS']
df['AVISTA_CUR90_DAYS'] = df['RES_EL_CUR90_DAYS'] + df['RES_GAS_CUR90_DAYS']
df['AVISTA_CUR_BAL_AMT'] = df['RES_EL_CUR_BAL_AMT'] + df['RES_GAS_CUR_BAL_AMT']

# All
df['TOTAL_30_DAYS_AMT'] = df['CITY_30_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR30_DAYS']
df['TOTAL_60_DAYS_AMT'] = df['CITY_60_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR60_DAYS']
df['TOTAL_90_DAYS_AMT'] = df['CITY_90_DAYS_PAST_DUE_AMT'] + df['AVISTA_CUR90_DAYS']
df['TOTAL_CUR_BALANCE'] = df['AVISTA_CUR_BAL_AMT'] + df['CITY_TOT_DUE']


# ## Generate Different Outcome Measures
# ### LAST_MO_W_DATA
# Last month with data on positive cases - estimate of when person started experiencing homelessness
# If have multiple ENROLL_DATEs, choose last month for each
# 
# ### 6_MO_PRIOR
# Within 6 months of last data month before experiencing homelessness?
# 
# ### MO_AWAY
# Number of months away from experiencing homelessness

# In[34]:


def get_outcomes(df):
    '''
    05/14/21
    Creates
        'LAST_MO_W_DATA' - boolean if P and last month with data
        'WITHIN_6_MO_PRIOR_LAST_DATA' - boolean if P and within 6 mo of last data
    '''
    new_df = df.copy()
    lasts = new_df[new_df.CMIS_MATCH].groupby(['SPA_PER_ID']).last()[['SPA_PREM_ID', 'MONTH']]
    lasts['LAST_MO_W_DATA'] = lasts['MONTH']
    lasts = lasts.reset_index().set_index(['SPA_PER_ID', 'SPA_PREM_ID']).drop('MONTH', axis=1)
    new_df = new_df.join(lasts, on=['SPA_PER_ID', 'SPA_PREM_ID'], how='left')
    # Create WITHIN_6_MO_PRIOR_LAST_DATA
    new_df['WITHIN_6_MO_PRIOR_LAST_DATA'] = (new_df['MONTH'] >= (new_df['LAST_MO_W_DATA'] - 6))
    # Create MO_AWAY
    new_df['MO_AWAY'] = new_df['LAST_MO_W_DATA'] - new_df['MONTH']
    # Change 'LAST_MO_W_DATA' to boolean
    new_df['LAST_MO_W_DATA'] = (new_df['LAST_MO_W_DATA'] == new_df['MONTH']).replace(to_replace=np.nan, value=False).astype('bool')
    return new_df


# In[35]:


df = get_outcomes(df)


# ## People and Premises
# * 'NUM_SPA_PER_ID_FOR_SPA_PREM_ID': number of people for each premises
# * 'NUM_SPA_PREM_ID_FOR_SPA_PER_ID': number of premises for each person

# In[36]:


def accumulate(df, grp_by_col, cumulative_col, new_col_name):
    '''
    05/02/21
    Finds cumulative counts.
    '''
    month_col = 'MONTH'
    cumulative = df[[month_col, grp_by_col, cumulative_col]].copy()
    # Find number of unique cumulateive elements
    cumulative = cumulative.drop_duplicates([grp_by_col, cumulative_col], keep='first').groupby([grp_by_col, month_col]).nunique()
    # Find cumulative count of unique elements
    cumulative[new_col_name] = (cumulative.groupby(grp_by_col)[cumulative_col].cumcount() + 1).astype('int64')
    cumulative.drop(cumulative_col, axis=1, inplace=True)
    # Join counts back to df
    new_df = df.join(cumulative, how='left', on=[grp_by_col, month_col])
    # Forward fill index gaps
    new_df[new_col_name].ffill(inplace=True)
    return new_df


# In[37]:


# Determine cumulative number of places a person has paid bills at so far
df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')

# Determine cumulative number of people a premesis has seen so far
df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')


# ## Size of MultiUnit¶
# number of SPA_PREM_ID's at same MULTI_DWELL_ID

# In[38]:


multi_dwell_size = df.groupby('MULTI_DWELL_ID').SPA_PREM_ID.nunique().rename('MULTI_DWELL_SIZE')
# Set size for not multi_unit to 0 
multi_dwell_size.loc[np.nan] = 0
df = df.join(multi_dwell_size, how='left', on='MULTI_DWELL_ID')


# # Get Preprocessing Stats and Save

# ## Print Columns

# In[39]:


df.columns.to_list()


# ## Check Nulls

# In[40]:


df.drop('MO_AWAY', axis=1).isnull().sum().sum()


# ## Save Pickle

filename = 'processed.pickle'
outfile = open(datapath+filename, 'wb')
pickle.dump(df, outfile)
outfile.close()


# ## Check Numbers Retained
retained_rows = len(df)
retained_accts = df.SPA_ACCT_ID.nunique()
retained_pos_cases = df[df.CMIS_MATCH].SPA_PER_ID.nunique()

print(f'Retained {retained_rows} = {100*retained_rows/rows}% of rows.')
print(f'Retained {retained_accts} = {100*retained_accts/accts}% of accounts.')
print(f'Retained {retained_pos_cases} = {100*retained_pos_cases/sa_pos_ppl}% of positive cases.')
print(f'People: {df.SPA_PER_ID.nunique()}')
print(f'Negative Cases: {df[~df.CMIS_MATCH].SPA_PER_ID.nunique()}')

# ## Total Time
print(calc_time.calc_time_from_sec(time.time()-startTime))

