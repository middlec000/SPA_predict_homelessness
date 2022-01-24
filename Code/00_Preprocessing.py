import pandas as pd
import numpy as np
import pickle
import time
from helper_methods import *

def main():
    filename = 'processed.pickle'
    datapath = '../Data/'

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
    df.MONTH = df.MONTH.apply(lambda x: date_map(date=x, relative_to=201512, format='yyyymm'))

    # Prepare for Matching
    df = df.drop_duplicates()

    df = df[~df.TOTAL_CUR_BALANCE.isna()]

    # Change NA for the following attributes to 0
    # Assume no data means there have been 0 occurrances of each of these
    df['BREAK_ARRANGEMENT'] = df['BREAK_ARRANGEMENT'].replace(to_replace=np.nan, value=0.0)
    df['PAST_DUE'] = df['PAST_DUE'].replace(to_replace=np.nan, value=0.0)

    # Just choose last of duplicates
    df = df.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

    # Get Service Agreements Data
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

    # DEAL WITH CMIS_MATCH
    # CMIS_MATCH: Replace NaN with False
    sa['CMIS_MATCH'] = sa['CMIS_MATCH'].replace(to_replace=np.nan, value=False).astype('bool')
    # If any CMIS_MATCH == True for person, then set CMIS_MATCH == True for all instances of person
    sa = sa.set_index('SPA_PER_ID')
    sa.update(sa.groupby('SPA_PER_ID')["CMIS_MATCH"].any())
    sa = sa.reset_index()
    sa = sa.drop_duplicates()

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

    """
    Retain only MAIN account holders
    Problem: Some people associated with many accounts AND some accounts have multiple people associated with them.

    Goal: Retain financial info closely related to each person. Also try to retain as many people as possible.

    Solution: Only retain main account holders - they are financially responsible for account.
    """
    sa = sa[sa['ACCT_REL_TYPE_CD'] == 'MAIN'].drop('ACCT_REL_TYPE_CD', axis=1)
    sa = sa.drop_duplicates()

    # Check Duplicates
    print(f"\nSA Duplicates? \n{sa.groupby(['SPA_ACCT_ID', 'SPA_PREM_ID']).size().value_counts()}")

    # Join to billing
    sa = sa.set_index(['SPA_ACCT_ID', 'SPA_PREM_ID'])
    df = df.join(sa, on=['SPA_ACCT_ID', 'SPA_PREM_ID'], how='inner')
    del sa

    """
    ## Fix Duplicate Identifiers
    Problem: Multiple instances of the same ('SPA_PER_ID', 'SPA_PREM_ID', 'MONTH') combination  

    Cause: Different accounts at same time - person switched accounts for some reason  

    Solution: Pick last
    """
    # Coerce id columns to ints
    to_ints = [
        'SPA_PER_ID',
        'SPA_PREM_ID',
        'SPA_ACCT_ID',
        'MONTH'
    ]
    for col in to_ints:
        df[col] = df[col].astype('int')

    df = df.groupby(['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']).last().reset_index()

    # Drop data that is post-ENROLL_DATE
    df = df[~(df['MONTH'] > df['ENROLL_DATE'])]
    df = df.drop('ENROLL_DATE', axis=1)

    # Create combined ID
    df['PER-PREM-MONTH_ID'] = df['SPA_PER_ID'].astype('str') + '-' + df['SPA_PREM_ID'].astype('str') + '-' + df['MONTH'].astype('str')

    # CREATE ADDITIONAL FEATURES
    # NUM_PREM_FOR_PER: cumulative number of places a person has paid bills at so far
    df = accumulate(df, grp_by_col='SPA_PER_ID', cumulative_col='SPA_PREM_ID', new_col_name='NUM_PREM_FOR_PER')

    # NUM_PER_FOR_PREM: cumulative number of people a premesis has seen so far
    df = accumulate(df, grp_by_col='SPA_PREM_ID', cumulative_col='SPA_PER_ID', new_col_name='NUM_PER_FOR_PREM')

    # Create interaction term PAST_DUE * NUM_PREM_FOR_PER TODO: Remove?
    # df['PAST_DUE*NUM_PREM_FOR_PER'] = df['PAST_DUE'] * df['NUM_PREM_FOR_PER']

    # Check nulls
    print(f"\nNulls: \n{df.isnull().sum().sum()}")

    # Check Grouping
    print(f"\nDuplicates? \n{df.groupby(['PER-PREM-MONTH_ID']).size().value_counts()}")

    # Check Data Types
    print(f'\nData Types: \n{df.dtypes}')

    # Save
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

    outfile = open(datapath+filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()

    # Print Time
    print(calc_time_from_sec(time.time()-startTime))
    return

if __name__ == "__main__":
    main()