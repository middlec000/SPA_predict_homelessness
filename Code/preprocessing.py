import pandas as pd
import numpy as np
import pickle
import time
from helper_methods import *

def main():
    filename = 'processed00.pickle'
    datapath = '../Data/'

    output = preprocess()
    outfile = open(datapath+filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    return

def preprocess(datapath: str):
    # Start Timer
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

    # Create combined billing attribute
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
    df.MONTH = df.MONTH.apply(lambda x: date_map(date=x, relative_to=201512, format='yyyymm'))
    print(f'Earliest Month: {df.MONTH.min()}')
    print(f'Latest Month: {df.MONTH.max()}')

    # Prepare for Matching
    df = df.drop_duplicates()

    df = df[~df.TOTAL_CUR_BALANCE.isna()]

    # Change NA for the following attributes to 0
    # Assume no data means there have been 0 occurrences of each of these
    df['BREAK_ARRANGEMENT'] = df['BREAK_ARRANGEMENT'].replace(to_replace=np.nan, value=0.0)
    df['PAST_DUE'] = df['PAST_DUE'].replace(to_replace=np.nan, value=0.0)

    # Just choose last of duplicates
    df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

    # Load Service Agreements Data
    sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').rename({
        'spa_prem_id':'SPA_PREM_ID', 
        'spa_acct_id':'SPA_ACCT_ID', 
        'spa_per_id':'SPA_PER_ID', 
        'homelessMatch':'CMIS_MATCH', 
        'EnrollDate':'ENROLL_DATE', 
        'apartment':'APARTMENT'
        }, axis=1)

    # Retain only columns we want to add to billing - note: all CMIS_MATCHes have ENROLL_DATEs
    sa_keep = [
        'SPA_PER_ID',
        'SPA_ACCT_ID',
        'SPA_PREM_ID',
        'ENROLL_DATE',
        'CMIS_MATCH',
        'ACCT_REL_TYPE_CD'
    ]
    sa = sa[sa_keep]

    """
    Problems:
    * Some accounts have multiple people associated with them at a time, some only have one
    * Some people are associated with multiple accounts (sometimes at different 'ACCT_REL_TYPE_CD')  

    Solution:  
    * Only retain the main account holder for each account
    """
    # Only keep info regarding the 'MAIN' account holder
    sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN']
    sa = sa.drop('ACCT_REL_TYPE_CD', axis=1).drop_duplicates()

    # DEAL WITH CMIS_MATCH
    # Replace NaN with False in CMIS_MATCH
    sa.CMIS_MATCH = sa.CMIS_MATCH.replace(to_replace=np.nan, value=False).astype('bool')
    # If any CMIS_MATCH == True for person, then set CMIS_MATCH == True for all instances of person
    sa = sa.set_index('SPA_PER_ID')
    sa.update(sa.groupby('SPA_PER_ID')["CMIS_MATCH"].any())
    # TODO: change back
    # sa.update(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID'])["CMIS_MATCH"].any())
    sa = sa.reset_index()
    sa = sa.drop_duplicates()

    # DEAL WITH ENROLL_DATE
    # Convert Dates to months since December, 2015
    sa['ENROLL_DATE'] = sa['ENROLL_DATE'].apply(lambda x: date_map(date=x, relative_to='2015-01-01', format='yyyy-mm-dd'))
    # Take min ENROLL_DATE
    enroll_dates = sa[~sa["ENROLL_DATE"].isnull()].groupby(["SPA_PER_ID"])["ENROLL_DATE"].min()
    sa = sa.set_index('SPA_PER_ID')
    sa.update(enroll_dates)
    sa = sa.reset_index()
    sa = sa.drop_duplicates()
    del enroll_dates

    # Collect Original Service Agreements Stats
    original_sa_stats = {
        'rows': len(sa),
        'accounts': sa['SPA_ACCT_ID'].nunique(),
        'premisses': sa['SPA_PREM_ID'].nunique(),
        'people': sa['SPA_PER_ID'].nunique(),
        'pos_people': sa[sa['CMIS_MATCH']]['SPA_PER_ID'].nunique(),
        'neg_people': sa[sa['CMIS_MATCH'] == False]['SPA_PER_ID'].nunique()
    }
    print(f'\nOriginal Service Agreement Stats\n{original_sa_stats}')

    # Check Matching
    print('\nService Agreement Grouping:')
    print(sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID']).size().value_counts())

    sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
    df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='inner')
    df = df.drop_duplicates()

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
    df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

    # Drop data that is post-ENROLL_DATE
    df = df[~(df['MONTH'] > df['ENROLL_DATE'])]
    df = df.drop('ENROLL_DATE', axis=1)

    # FEATURE ENGINEERING
    # Create Combined ID
    df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')

    # Cumulative number of places a person has paid bills at for each month
    df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')

    # Cumulative number of people a premesis has seen for each month
    df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')

    # CHECK RESULTS
    # Check nulls
    print(f"\nNulls: \n{df.isnull().sum().sum()}")

    # Check Grouping
    print(f"\nDuplicates? \n{df.groupby(['PER-PREM-MONTH_ID']).size().value_counts()}")

    # Check Data Types
    print(f'\nData Types: \n{df.dtypes}')

    # Create Output
    output = {
        'Data': df,
        'Features': df.columns.to_list(),
        'Data_Retention_Stats': generate_log(
            original_billing = original_billing_stats,
            original_sa = original_sa_stats,
            final_df = df
            )
    }

    print('\nData Retention Stats:')
    print_dict(output["Data_Retention_Stats"])

    """
    # Save
    outfile = open(datapath+filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    """

    print("Preprocessing Time:")
    print(calc_time_from_sec(time.time()-startTime))
    return output

if __name__ == '__main__':
    main()