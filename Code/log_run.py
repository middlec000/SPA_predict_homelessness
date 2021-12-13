import pandas as pd
import pickle
import time
from log_helper_methods import *

def main():
    startTime = time.time()

    datapath = '../Data/'

    save_filename = 'bill'

    filename = datapath+'processed.pickle'
    infile = open(filename,'rb')
    df = pickle.load(infile)
    infile.close()

    #for column in df.columns.to_list():
    #    print(column)

    keep = [
        # 'PAST_DUE',
        # 'TOTAL_60_DAYS_AMT',
        'TOTAL_CUR_BALANCE',
        # 'NUM_PREM_FOR_PER',
        # 'BREAK_ARRANGEMENT',
        # 'MULTI_DWELL_SIZE',
        # 'SNAP_GEO',
        # 'NUM_PER_FOR_PREM',
        # 'HAS_COTENANT',
        'PER-PREM-MONTH_ID',
        'CMIS_MATCH',
    ]

    df = df[keep]
    print('Reduced Data...Done')
    print(calc_time_from_sec(time.time() - startTime))

    # Model Parameters
    event_col = 'CMIS_MATCH'
    id_col = 'PER-PREM-MONTH_ID'
    model = 'logit'
    scaler = None
    sampler = None
    k = 4
    #scaler = StandardScaler()
    #sampler = RandomOverSampler(sampling_strategy='minority', random_state=42)

    df, models = log_k_folds(
        df = df, 
        event_col = event_col,
        id_col = id_col, 
        k = k, 
        model = model, 
        scaler = scaler,
        sampler = sampler,
    )
    print('K-Folds...Done')
    print(calc_time_from_sec(time.time() - startTime))

    # ## Extract IDs
    df[['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']] = df['PER-PREM-MONTH_ID'].str.split(pat='-', expand=True)

    df = df.drop('PER-PREM-MONTH_ID', axis=1)

    for col in ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']:
        df[col] = df[col].astype('int')

    # ## Take Max Likelihood For Each Person
    predictions = df.groupby('SPA_PER_ID')['prediction'].max()
    predictions = pd.concat([predictions, df.groupby('SPA_PER_ID').CMIS_MATCH.any()], axis=1, join='inner', ignore_index=False)
    predictions = predictions.reset_index()

    # Get metrics at each possible threshold
    summary = get_metrics(df=predictions, y_true='CMIS_MATCH', y_pred='prediction')
    print('Get Summary Metrics...Done')
    print(calc_time_from_sec(time.time() - startTime))

    output = {'Performance': summary, 'Models': models, 'Features': keep}

    # save output
    outfile = open(datapath + 'output_'+save_filename+'.pickle', 'wb')
    pickle.dump(output, outfile)
    outfile.close()

    print('\nTotal Time:')
    print(calc_time_from_sec(time.time() - startTime))
    return

if __name__ == "__main__":
    main()