import pandas as pd
import numpy as np
import pickle
import time
from custom_methods.calc_time import calc_time_from_sec

startTime = time.time()
datapath = '../Data/'

# # Load Data
filename = 'log_ready.pickle'
infile = open(datapath+filename,'rb')
df = pickle.load(infile)
infile.close()
print(df.columns.to_list())

# # K-Folds

# ## Helper Programs
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
    10/26/21
    Chooses k-folds based on 'id_col'
    Supported use of the following models:
        statsmodels.discrete.discrete_model.Logit
            set model='logit'
    
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
            X_res, y_res = fold_sampler.fit_sample(X=df_train.drop(event_col, axis=1), y=df_train[event_col])
            df_train = pd.concat([X_res, y_res], axis=1)
            
        # Get predictions
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

        # Append to predictions, actuals
        id_prediction_actual = id_prediction_actual.append(pd.concat([
            df_test[id_col].reset_index(drop=True), \
            predictions.rename('prediction').reset_index(drop=True), \
            df_test[event_col].reset_index(drop=True)], 
            axis=1),ignore_index=True)
    id_prediction_actual[event_col] = id_prediction_actual[event_col].astype('bool')
    return id_prediction_actual, models


# ## Run
#from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import RandomOverSampler

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


# ## Print Predictions Head
print(predictions.head())

# ## Print Sample Model Fitting
print(models[0].summary())

# # Save and Time
# Save predictions
filename = 'log_output.pickle'
outfile = open(datapath + filename, 'wb')
pickle.dump(predictions, outfile)
outfile.close()

# Save models
filename = 'log_models.pickle'
outfile = open(datapath + filename, 'wb')
pickle.dump(models, outfile)
outfile.close()

print(calc_time_from_sec(time.time() - startTime))

