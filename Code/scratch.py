import pandas as pd

datapath = '../Data/'

sa = pd.read_csv(datapath+'ServiceAgreements_Anon.csv').\
    rename({'spa_prem_id':'SPA_PREM_ID', 'spa_acct_id':'SPA_ACCT_ID', 'spa_per_id':'SPA_PER_ID', 'homelessMatch':'CMIS_MATCH', 'EnrollDate':'ENROLL_DATE', 'apartment':'APARTMENT'}, axis=1)

sa = sa.iloc[:200]

