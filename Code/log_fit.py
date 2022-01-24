import pandas as pd
import pickle
import time
from helper_methods import *

def main():
    input_filename = 'processed00.pickle'
    save_filename  = 'output00.pickle'

    datapath = '../Data/'

    filename = datapath+input_filename
    infile = open(filename,'rb')
    preprocessed = pickle.load(infile)
    infile.close()

    output = log_fit(preprocessed=preprocessed)

    # save output
    outfile = open(datapath + save_filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    return

def log_fit(preprocessed):
    startTime = time.time()
   
    # input_filename = 'processed00.pickle'
    # save_filename  = 'output00.pickle'

    """datapath = '../Data/'

    filename = datapath+input_filename
    infile = open(filename,'rb')
    preprocessed = pickle.load(infile)
    infile.close()"""

    df = preprocessed['Data']
    features = preprocessed['Features']
    data_retention_stats = preprocessed['Data_Retention_Stats']
    # del preprocessed

    print('Features:')
    print_list(features)
    print('\nData_Retention_Stats:')
    print_dict(data_retention_stats)

    """datapath = '../Data/'
    filename = datapath+input_filename
    infile = open(filename,'rb')
    df = pickle.load(infile)
    infile.close()"""

    print(df.head())

    keep = [
        'PAST_DUE',
        # 'TOTAL_60_DAYS_AMT',
        'TOTAL_CUR_BALANCE',
        'NUM_PREM_FOR_PER',
        'BREAK_ARRANGEMENT',
        # 'MULTI_DWELL_SIZE',
        # 'SNAP_GEO',
        'NUM_PER_FOR_PREM',
        # 'HAS_COTENANT',
        'PER-PREM-MONTH_ID',
        # 'PAST_DUE*NUM_PREM_FOR_PER',
        'CMIS_MATCH',
    ]

    print('\nRetained Features:')
    print_list(keep)

    df = df[keep]

    # Model Parameters
    event_col = 'CMIS_MATCH'
    id_col = 'PER-PREM-MONTH_ID'
    model = 'logit'
    scaler = None
    sampler = None
    k = 4
    #scaler = StandardScaler()
    #sampler = RandomOverSampler(sampling_strategy='minority', random_state=42)

    predictions, models = log_k_folds(
        df = df, 
        event_col = event_col,
        id_col = id_col, 
        k = k, 
        model = model, 
        scaler = scaler,
        sampler = sampler,
    )
    print('\nK-Folds...Done')
    print(calc_time_from_sec(time.time() - startTime))

    # ## Extract IDs
    predictions[['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']] = predictions['PER-PREM-MONTH_ID'].str.split(pat='-', expand=True)

    predictions = predictions.drop('PER-PREM-MONTH_ID', axis=1)

    for col in ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']:
        predictions[col] = predictions[col].astype('float').astype('int')

    # ## Take Max Predicted Likelihood For Each Person
    predictions_grp = predictions.groupby('SPA_PER_ID')['prediction'].max()
    predictions = pd.concat([predictions_grp, predictions.groupby('SPA_PER_ID')['CMIS_MATCH'].any()], axis=1, join='inner', ignore_index=False)
    predictions = predictions.reset_index()
    del predictions_grp

    # Get metrics at each possible threshold
    summary = get_metrics(df=predictions, y_true='CMIS_MATCH', y_pred='prediction')
    print('\nGet Summary Metrics...Done')
    print(calc_time_from_sec(time.time() - startTime))

    output = {
        'Performance': summary,
        'Models': models,
        'Features': keep,
        'Data_Retention_Stats': data_retention_stats
        }

    """# save output
    outfile = open(datapath + save_filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()"""

    print('\nTotal Time:')
    print(calc_time_from_sec(time.time() - startTime))
    return output

if __name__ == "__main__":
    main()