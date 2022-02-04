import pandas as pd
import pickle
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
    df = preprocessed['Data']
    features = preprocessed['Features']
    data_retention_stats = preprocessed['Data_Retention_Stats']

    keep = [
        'PAST_DUE',
        'TOTAL_CUR_BALANCE',
        'NUM_PREM_FOR_PER',
        'BREAK_ARRANGEMENT',
        'NUM_PER_FOR_PREM',
        'PER-PREM-MONTH_ID',
        'CMIS_MATCH',
    ]

    df = df[keep]

    # Run K-Folds with Logistic Regression
    predictions, models = log_k_folds(
        df = df, 
        event_col = 'CMIS_MATCH',
        id_col = 'PER-PREM-MONTH_ID', 
        k = 4, 
        model = 'logit', 
        scaler = None,
        sampler = None,
    )

    # ## Extract IDs
    predictions[['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']] = predictions['PER-PREM-MONTH_ID'].str.split(pat='-', expand=True)
    predictions = predictions.drop('PER-PREM-MONTH_ID', axis=1)

    # Convert Data Types
    for col in ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']:
        predictions[col] = predictions[col].astype('float').astype('int')

    # ## Take Max Predicted Likelihood For Each Person
    predictions_grp = predictions.groupby('SPA_PER_ID')['prediction'].max()
    predictions = pd.concat([predictions_grp, predictions.groupby('SPA_PER_ID')['CMIS_MATCH'].any()], axis=1, join='inner', ignore_index=False)
    predictions = predictions.reset_index()
    del predictions_grp

    # Get metrics at each threshold
    summary = get_metrics(df=predictions, y_true='CMIS_MATCH', y_pred='prediction')

    # Generate output
    output = {
        'Performance': summary,
        'Models': models,
        'Features': keep,
        'Data_Retention_Stats': data_retention_stats
        }
    return output

if __name__ == "__main__":
    main()