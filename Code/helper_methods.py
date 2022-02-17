import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import auc, roc_curve
from typing import Union

def get_folds_ids(id_col, k):
    # 04/24/21
    all_ids = pd.Series(id_col.unique().copy(), dtype='object')
    folds = []
    num_per_fold = int(np.ceil(len(all_ids)/k))
    while not all_ids.empty:
        fold = []
        if len(all_ids) < num_per_fold:
            fold = all_ids
        else:
            fold = all_ids.sample(n=num_per_fold)
        folds.append(fold.to_list())
        all_ids.drop(index=fold.index, inplace=True)
    return np.array(folds, dtype='object')


def log_k_folds(df: pd.DataFrame, event_col: str, id_col: str, k: int, model: object, sampler: object=None, scaler: object=None, inverted:bool=False) -> pd.DataFrame:
    '''
    2/17/22
    Chooses k-folds based on 'id_col'
    Supported use of the following models:
        statsmodels.discrete.discrete_model.Logit
            set model='logit'
    Supported use of the following sampling methods:
        imblearn.over_sampling.RandomOverSampler
        imblearn.under_sampling.RandomUnderSampler
    
    Returns:
        predictions (pandas DataFrame), models (List[statsmodels.discrete.discrete_model.Logit])
    '''
    import statsmodels.api as sm
    import copy

    # Create copies of base model to train
    models = []
    for i in range(k):
        models.append(copy.deepcopy(model))
    # Create Folds
    folds = get_folds_ids(id_col=df[id_col], k=k)
    id_prediction_actual = pd.DataFrame(columns=[id_col, 'prediction', event_col])
    # Loop through k-folds
    for fold in range(len(folds)):
        # Find test, training sets for fold
        if inverted:
            # Training set is smaller than test set
            df_test = df[~df[id_col].isin(folds[fold])]
            df_train = df[df[id_col].isin(folds[fold])]
        else:
            df_test = df[df[id_col].isin(folds[fold])]
            df_train = df[~df[id_col].isin(folds[fold])]
            
        # Scale
        if scaler != None:
            fold_scaler = copy.deepcopy(scaler)
            df_train.update(fold_scaler.fit_transform(df_train.drop([event_col, id_col], axis=1)))
            df_test.update(fold_scaler.transform(df_test.drop([event_col, id_col], axis=1)))
            
        # Undersample, oversample, etc.
        if sampler != None:
            fold_sampler = copy.deepcopy(sampler)
            X_res, y_res = fold_sampler.fit_resample(X=df_train.drop(event_col, axis=1), y=df_train[event_col])
            df_train = pd.concat([X_res, y_res], axis=1)
            
        # Fit Model and Get predictions
        predictions = None
        if model == "logit":           
            exog = np.asarray(df_train.drop([id_col, event_col], axis=1), dtype='float')
            endog = np.asarray(df_train[event_col], dtype='bool')
            model_local = sm.Logit(endog=endog, exog=exog).fit(disp=False)
            models[fold] = model_local
            exog_test = np.asarray(df_test.drop([id_col, event_col], axis=1), dtype='float')
            predictions = pd.Series(model_local.predict(exog=exog_test))
        else:
            print("Model not supported.")
            return

        # Append to predictions, actuals
        id_prediction_actual = id_prediction_actual.append(pd.concat([
            df_test[id_col].reset_index(drop=True), \
            predictions.rename('prediction').reset_index(drop=True), \
            df_test[event_col].reset_index(drop=True)], 
            axis=1),ignore_index=True)
    id_prediction_actual[event_col] = id_prediction_actual[event_col].astype('bool')
    return id_prediction_actual, models

def get_metrics(df: pd.DataFrame, y_true: str, y_pred: str) -> pd.DataFrame:
    # Overall Counts
    total = len(df)
    p = df[y_true].sum()
    n = total - p
    # True Positive Rate (TPR) aka Sensitivity aka Recall aka Hit Rate
    # False Positive Rate (FPR) aka False Alarm Rate
    fpr, tpr, threshold = roc_curve(y_true=df[y_true].astype('bool'), y_score=df[y_pred])
    data = {'threshold': threshold, 'fpr': fpr, 'tpr': tpr}
    summary = pd.DataFrame(data=data)
    # Basic Measures
    summary['tp'] = summary['tpr'] * p
    summary['fp'] = summary['fpr'] * n
    summary['tn'] = n - summary['fp']
    summary['fn'] = p - summary['tp']
    # True Negative Rate (TNR)/Specificity
    summary['tnr'] = summary['tn'] / n
    # Balanced Accuracy
    summary['balanced'] = (summary['tpr'] + summary['tnr']) / 2
    # Positive Predictive Value (PPV)/Precision
    summary['ppv'] = summary['tp'] / (summary['tp'] + summary['fn'])
    # Negative Predictive Value (NPV)
    summary['npv'] = summary['tn'] / (summary['tn'] + summary['fp'])
    # F-1
    summary['f-1'] = 2 * summary['ppv'] * summary['tpr'] / (summary['ppv'] + summary['tpr'])
    # Accuracy
    summary['accuracy'] = (summary['tp'] + summary['tn']) / (p + n)
    # Area Under (ROC) Curve (AUC)
    summary['auc'] = auc(x=summary['fpr'], y=summary['tpr'])
    return summary

def calc_time_from_sec(seconds: float) -> None:
    '''
    05/06/21
    prints time as hours:minutes:seconds
    '''
    sec = seconds
    mn = (sec - sec%60)/60
    sec = sec-60*mn
    hr = int((mn - mn%60)/60)
    mn = int(mn - 60*hr)
    print(f"hours:minutes:seconds = {hr}:{mn}:{sec}")
    return

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

def generate_log(original_billing: dict, original_sa: dict, final_df: pd.DataFrame) -> dict:
    """Generates log for preprocessing step.

    Args:
        original_billing (dict): [description]
        original_sa (dict): [description]
        final_df (pd.DataFrame): [description]

    Returns:
        dict: [description]
    """
    retained_rows = len(final_df)
    retained_accts = final_df.SPA_ACCT_ID.nunique()
    retained_premisses = final_df.SPA_PREM_ID.nunique()
    retained_ppl = final_df.SPA_PER_ID.nunique()
    retained_pos_cases = final_df[final_df.CMIS_MATCH].SPA_PER_ID.nunique()
    retained_neg_cases = final_df[~final_df.CMIS_MATCH].SPA_PER_ID.nunique()
    retention_stats = {
        'Rows': retained_rows,
        'Rows (% of Billing)': 100*retained_rows/original_billing['rows'],
        'Accounts': retained_accts,
        'Accounts (% of Billing)': 100*retained_accts/original_billing['accounts'],
        'Premisses': retained_premisses,
        'Premisses (% of Billing)': 100*retained_premisses/original_billing['premisses'],
        'People': retained_ppl,
        'People (% of Service Agreements)': 100*retained_ppl/original_sa['people'],
        'Positive Cases': retained_pos_cases,
        'Positive Cases (% of Service Agreements)': 100*retained_pos_cases/original_sa['pos_people'],
        'Negative Cases': retained_neg_cases,
        'Negative Cases (% of Service Agreements)': 100*retained_neg_cases/original_sa['neg_people'],
    }
    return retention_stats

def print_list(list_to_print: list):
    for item in list_to_print:
        print(item)
    return

def print_dict(dict_to_print: dict):
    for k, v in dict_to_print.items():
        print(f'{k}: {v}')
    return

def date_map(date: object, relative_to: object, format: str) -> Union[int,float]:
    '''
    05/06/21
    Map date to number of months since 'relative_to' month.
    'relative_to' must be in same format as 'date'.

    Supported formats:
        'yyyy-mm-dd'
        'yyyymm'
        'yyyy'
    '''
    relative_year = None
    relative_month = None
    year = None
    month = None
    # If 'date' is NaN or None, return NaN
    if (type(date) == float) or (date is None):
        if np.isnan(date) or (date is None):
            return np.nan
    # Otherwise carry out formatting
    if format.lower() == 'yyyy-mm-dd':
        relative_year = int(str(relative_to)[0:4])
        relative_month = int(str(relative_to)[5:7])
        year = int(str(date)[0:4])
        month = int(str(date)[5:7])
    elif format.lower() == 'yyyymm':
        relative_year = int(str(relative_to)[0:4])
        relative_month = int(str(relative_to)[4:6])
        year = int(str(date)[0:4])
        month = int(str(date)[4:6])
    elif format.lower() == 'yyyy':
        relative_year = int(relative_to)
        year = int(date)
    else:
        print('Unsupported date format.')
        return
    
    if month is None:
        return int((year - relative_year)*12)
    else:
        return int((year - relative_year)*12 + (month - relative_month))

def calc_time_from_sec(seconds: float) -> None:
    '''
    05/06/21
    prints time as hours:minutes:seconds
    '''
    sec = seconds
    mn = (sec - sec%60)/60
    sec = sec-60*mn
    hr = int((mn - mn%60)/60)
    mn = int(mn - 60*hr)
    print(f"hours:minutes:seconds = {hr}:{mn}:{sec}")
    return

def load_model_output(filepath: str):
    """
    01/23/22
    Load output from model fitting.

    Args:
        filepath (str): Filepath of model output pickle file.

    Returns:
        [type]: performance, models, features, data_retention_stats
    """
    infile = open(filepath, 'rb')
    model_output = pickle.load(infile)
    infile.close()
    return model_output['Performance'], model_output['Models'], model_output['Features'], model_output['Data_Retention_Stats']