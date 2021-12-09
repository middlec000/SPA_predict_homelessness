import pandas as pd
import pickle
import time
from custom_methods.calc_time import calc_time_from_sec

start_time = time.time()

datapath = '../Data/'

filename = datapath+'processed.pickle'
infile = open(filename,'rb')
df = pickle.load(infile)
infile.close()

for column in df.columns.to_list():
    print(column)

keep = [
    'PAST_DUE',
    # 'TOTAL_60_DAYS_AMT',
    'TOTAL_CUR_BALANCE',
    'NUM_PREM_FOR_PER',
    'BREAK_ARRANGEMENT',
    # 'COVID_REMINDER',
    'MULTI_DWELL_SIZE',
    'SNAP_GEO',
    'NUM_PER_FOR_PREM',
    # 'APARTMENT', convergence issues
    'HAS_COTENANT',
    'PER-PREM-MONTH_ID',
    'CMIS_MATCH',
]

df = df[keep]

# # Clean Up
df = df.drop_duplicates()
df.isnull().sum().sum()

"""# # Save and Time
filename = 'log_ready.pickle'
outfile = open(datapath+filename, 'wb')
pickle.dump(df, outfile)
outfile.close()"""

print(calc_time_from_sec(time.time()-start_time))

