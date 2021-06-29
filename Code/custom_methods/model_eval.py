import pandas as pd
import numpy as np
import copy

def evaluate_model(binned_predictions: pd.Series, actuals: pd.Series, decimals=6) -> pd.DataFrame:
    '''
    04/25/21
    Note: binned_predictions and actuals must both have same index.
    Custom is harmonic mean of precision_p and recall_n (for P), precision_n and recall_p (for N).
    '''
    if len(binned_predictions) != len(actuals):
        print("Lengths are not equal.")
        return None
    if (binned_predictions.dtype != 'bool') or (actuals.dtype != 'bool'):
        print("Must be boolean Series objects.")
        return None
    idx = pd.Index(['Positive', 'Negative'], name='Case')
    results = pd.DataFrame(index=idx, columns=['Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom'])
    # true positives
    tp = float((binned_predictions & actuals).sum())
    # true negatives
    tn = float((~binned_predictions & ~actuals).sum())
    # false positives
    fp = float((binned_predictions & ~actuals).sum())
    # false negatives
    fn = float((~binned_predictions & actuals).sum())
    # print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn}")
    try:
        p_prec = tp/(tp+fp)
        n_prec = tn/(tn+fn)
        p_rec = tp/(tp+fn)
        n_rec = tn/(tn+fp)
        results.loc['Positive'] = {
            'Number Predicted':tp+fp, 
            'Actual Number':tp+fn, 
            'False P/N Rate':fp/(fp+tn), 
            'Precision':p_prec, 
            'Recall':p_rec, 
            'f-1':tp/(tp+(1/2)*(fp+fn)), 
            #'Custom':2*p_prec*n_rec/(p_prec+n_rec)
            'Custom':(1/2)*p_prec + (1/2)*p_rec
            }
        results.loc['Negative'] = {
            'Number Predicted':tn+fn, 
            'Actual Number':tn+fp, 
            'False P/N Rate':fn/(fn+tp), 
            'Precision':n_prec,
            'Recall':n_rec, 
            'f-1':tn/(tn+(1/2)*(fp+fn)), 
            #'Custom':2*n_prec*p_rec/(n_prec+p_rec)
            'Custom':(1/2)*n_prec + (1/2)*n_rec
            }
        results['Number Predicted'] = results['Number Predicted'].astype('int64')
        results['Actual Number'] = results['Actual Number'].astype('int64')
    except:
        return None
    pd.options.display.float_format = ("{:,."+str(decimals)+"f}").format
    return results.sort_index(level='Case', ascending=False)

def get_model_metrics(predictions: pd.Series, actuals: pd.Series, cutoffs: list=[0.01,0.05,0.10]) -> pd.DataFrame:
    '''
    05/03/21
    cutoffs: can be list of proportions or numbers (or mixed)
        will classify top proportion or number based on predicted risk as Positive
    Returns:
        index levels = [[], ['Positive', 'Negative']]
        columns = ['Threshold_Value', 'Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    '''
    if len(predictions) != len(actuals):
        print("Lengths don't match!")
        return
    # Get threshold numbers from cutoffs
    top_thresholds = []
    sorted_predictions = predictions.sort_values(ascending=False)
    # Also create outer_index for summary dataframe
    for cutoff in cutoffs:
        top_number = cutoff
        if int(top_number) == 0:
            # cutoff must be proportion
            top_number = round(len(predictions) * cutoff)
        top_threshold = sorted_predictions.iloc[top_number]
        top_thresholds.append(top_threshold)
    # Reset index on actuals
    actuals_reset = actuals.reset_index(drop=True).astype('bool')
    # Summary dataframe to store results
    idx = pd.MultiIndex.from_product([cutoffs, ['Positive', 'Negative']], names=['Threshold', 'Case'])
    columns = ['Threshold_Value', 'Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    summary = pd.DataFrame(index=idx, columns=columns)
    for i in range(len(cutoffs)):
        binned_predictions = (predictions >= top_thresholds[i])
        evaluation = evaluate_model(binned_predictions, actuals_reset)
        if not (evaluation is None):
            summary.loc[cutoffs[i], 'Threshold_Value'] = top_thresholds[i]
            summary.loc[(cutoffs[i], "Positive"), 'Number Predicted':'Custom'] = evaluation.loc["Positive"]
            summary.loc[(cutoffs[i], "Negative"), 'Number Predicted':'Custom'] = evaluation.loc["Negative"]
    return summary.sort_index(level=['Threshold', 'Case'], ascending=[True,False])

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

def split_on_people(df: pd.DataFrame, id_col: str, test_size: int=None, test_frac: float=0.33):
    '''
    05/05/21
    Returns:
        df_train, df_test
    '''
    people = pd.Series(df[id_col].unique())
    if test_size == None:
        test_people = people.sample(frac=test_frac)
    else:
        test_people = people.sample(n=test_size)
    train_people = people[~people.isin(test_people)]
    return df[df[id_col].isin(train_people)], df[df[id_col].isin(test_people)]

def k_fold_models(df: pd.DataFrame, event_col: str, id_col: str, k: int, model: object, cutoffs: list=[0.01,0.05,0.10], sampler: object=None, scaler: object=None) -> pd.DataFrame:
    '''
    05/18/21
    Chooses k-folds based on 'id_col'
    Supported use of the following models:
        sklearn.linear_model._logistic.LogisticRegression
        statsmodels.discrete.discrete_model.Logit
            set model='logit'
        lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter
        tensorflow.python.keras.engine.sequential.Sequential
    
    Returns:
        results, summary
        
    To get list of models:
        models = results.index.get_level_values(level='Model')
    '''
    import statsmodels.api as sm

    # Create copies of base model to train
    models = []
    for i in range(k):
        models.append(copy.deepcopy(model))
    # Create results dataframe to track stuff
    columns = ['Threshold_Value', 'Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    idx = pd.MultiIndex.from_product([np.arange(0,k,1), cutoffs, ['Positive', 'Negative']], names=['Fold', 'Threshold', 'Case'])
    results = pd.DataFrame(index=idx, columns=columns)
    # Create Folds
    folds = get_folds_ids(id_col=df[id_col], k=k)
    # Loop through k-folds
    for fold in results.index.get_level_values('Fold').unique():
        # Find test, training sets for fold
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
        if str(type(model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>":
            models[fold].fit(X=df_train.drop([event_col, id_col], axis=1), y=df_train[event_col])
            predictions = pd.Series(np.transpose(models[fold].predict_proba(df_test.drop([event_col, id_col], axis=1)))[1], dtype='float')

        elif model == "logit":
            exog = np.asarray(sm.add_constant(df_train.drop([id_col, event_col], axis=1)), dtype='float')
            endog = np.asarray(df_train[event_col], dtype='bool')
            model_local = sm.Logit(endog=endog, exog=exog).fit(disp=False)
            models[fold] = model_local
            exog_test = np.asarray(sm.add_constant(df_test.drop([id_col, event_col], axis=1)), dtype='float')
            predictions = pd.Series(model_local.predict(exog=exog_test))

        elif str(type(model)) == "<class 'lifelines.fitters.coxph_fitter.CoxPHFitter'>":
            '''
            # Cumulative Hazard - Time Dependence
            cutoff_month = 6
            models[fold].fit(df_train.drop(id_col, axis=1), duration_col='DURATION', event_col=event_col, step_size=0.1)
            predictions = models[fold].predict_cumulative_hazard(df_test.drop(event_col, axis=1)).loc[cutoff_month]
            '''
            # Partial Hazard - No Time Dependence
            models[fold].fit(df_train.drop(id_col, axis=1), duration_col='DURATION', event_col=event_col, step_size=0.1)
            predictions = models[fold].predict_partial_hazard(df_test.drop(event_col, axis=1))
            print(f"fold: {fold}")
            print(f"predictions: {len(predictions)}\n{predictions.head()}")
            print(f"acutals: {len(df_test)}\n{df_test.head()[event_col]}\n")
            

        elif str(type(model)) == "<class 'lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter'>":
            models[fold].fit(df=df_train, event_col=event_col, start_col='start', stop_col='stop', id_col=id_col, step_size=0.1)
            predictions = models[fold].predict_partial_hazard(df_test.drop(event_col, axis=1))

        elif str(type(model)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
            models[fold].fit(df_train.drop([event_col, id_col], axis=1), df_train[event_col], batch_size=10, epochs=5)
            predictions = models[fold].predict(x=df_test.drop([event_col, id_col], axis=1))
            predictions = pd.Series(np.transpose(predictions)[0])

        # Get model evaluation
        model_metrics = get_model_metrics(predictions=predictions, actuals=df_test[event_col], cutoffs=cutoffs)
        #print(model_metrics)
        results.loc[fold].update(model_metrics)
    # Create summary of results
    results = results.astype('float').sort_index(level=['Fold', 'Threshold', 'Case'], ascending=[True,True,False])
    summary = results.groupby(level=['Threshold','Case']).mean().sort_index(level=['Threshold', 'Case'], ascending=[True,False])
    results.index = results.index.set_levels(models, level='Fold').rename('Model', level='Fold')
    
    return results, summary

def k_folds(df: pd.DataFrame, event_col: str, id_col: str, k: int, model: object, sampler: object=None, scaler: object=None, inverted:bool=False) -> pd.DataFrame:
    '''
    05/21/21
    Chooses k-folds based on 'id_col'
    Supported use of the following models:
        sklearn.linear_model._logistic.LogisticRegression
        statsmodels.discrete.discrete_model.Logit
            set model='logit'
        lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter
        tensorflow.python.keras.engine.sequential.Sequential
    
    Returns:
        predictions
        
    To get list of models:
        models = results.index.get_level_values(level='Model')
    '''
    import statsmodels.api as sm

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
        if str(type(model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>":
            models[fold].fit(X=df_train.drop([event_col, id_col], axis=1), y=df_train[event_col])
            predictions = pd.Series(np.transpose(models[fold].predict_proba(df_test.drop([event_col, id_col], axis=1)))[1], dtype='float')

        elif model == "logit":
            #exog = np.asarray(sm.add_constant(df_train.drop([id_col, event_col], axis=1)), dtype='float')
            exog = np.asarray(df_train.drop([id_col, event_col], axis=1), dtype='float')
            endog = np.asarray(df_train[event_col], dtype='bool')
            model_local = sm.Logit(endog=endog, exog=exog).fit(disp=False)
            models[fold] = model_local
            #exog_test = np.asarray(sm.add_constant(df_test.drop([id_col, event_col], axis=1)), dtype='float')
            exog_test = np.asarray(df_test.drop([id_col, event_col], axis=1), dtype='float')
            predictions = pd.Series(model_local.predict(exog=exog_test))

        elif str(type(model)) == "<class 'lifelines.fitters.coxph_fitter.CoxPHFitter'>":
            # Partial Hazard - No Time Dependence
            models[fold].fit(df_train.drop(id_col, axis=1), duration_col='DURATION', event_col=event_col, step_size=0.1)
            predictions = models[fold].predict_partial_hazard(df_test.drop(event_col, axis=1))

        elif str(type(model)) == "<class 'lifelines.fitters.cox_time_varying_fitter.CoxTimeVaryingFitter'>":
            models[fold].fit(df=df_train, event_col=event_col, start_col='start', stop_col='stop', id_col=id_col, step_size=0.1)
            predictions = models[fold].predict_partial_hazard(df_test.drop(event_col, axis=1))

        elif str(type(model)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
            models[fold].fit(df_train.drop([event_col, id_col], axis=1), df_train[event_col], batch_size=10, epochs=5)
            predictions = models[fold].predict(x=df_test.drop([event_col, id_col], axis=1))
            predictions = pd.Series(np.transpose(predictions)[0])

        # Append to predictions, actuals
        id_prediction_actual = id_prediction_actual.append(pd.concat([
            df_test[id_col].reset_index(drop=True), \
            predictions.rename('prediction').reset_index(drop=True), \
            df_test[event_col].reset_index(drop=True)], 
            axis=1),ignore_index=True)
    id_prediction_actual[event_col] = id_prediction_actual[event_col].astype('bool')
    return id_prediction_actual

def get_summary(df:pd.DataFrame, event_col:str='CMIS_MATCH'):
    '''
    05/20/21
    Finds max values over positive versions of metrics
    df must have columns: ['prediction', event_col]
    '''
    columns = ['Threshold_Value', 'Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    idx = pd.MultiIndex.from_product([['Precision', 'Recall', 'f-1', 'Custom'], ['Positive', 'Negative']], names=['Max_Of', 'Case'])
    summary = pd.DataFrame(index=idx, columns=columns)

    sorted_predictions = df.sort_values(by='prediction')
    for threshold in sorted_predictions['prediction']:
        binned_predictions = (df['prediction'] > threshold)
        current_summary = evaluate_model(binned_predictions=binned_predictions, actuals=df[event_col].astype('bool'))
        if not (current_summary is None):
            for metric in summary.index.get_level_values('Max_Of').unique():
                if (summary.loc[(metric, 'Positive'), metric] < current_summary.loc['Positive', metric]) or np.isnan(summary.loc[(metric, 'Positive'), metric]):
                    summary.loc[(metric, 'Positive'), 'Number Predicted':'Custom'] = current_summary.loc['Positive']
                    summary.loc[(metric, 'Negative'), 'Number Predicted':'Custom'] = current_summary.loc['Negative']
                    summary.loc[metric, 'Threshold_Value'] = threshold
    int_cols = ['Number Predicted', 'Actual Number']
    float_cols = ['Threshold_Value', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    for col in int_cols:
        summary[col] = summary[col].astype('int')
    for col in float_cols:
        summary[col] = summary[col].astype('float')
    return summary

def print_summary(frame:pd.DataFrame, float_format:str='%.2f'):
    summary_print = frame.copy()
    summary_print['Actual Number'] = (summary_print['Number Predicted'] * summary_print['Precision']).astype('int')
    summary_print = summary_print.reset_index().drop(['Threshold_Value', 'Custom'], axis=1).rename(
        {'Number Predicted':'Predicted', 'Actual Number':'Hits', 'False P/N Rate':'False P/N', 'Case':'Class'}, axis=1)
    summary_print['Class'] = summary_print['Class'].apply(lambda x: 'P' if x=='Positive' else 'N')
    summary_print = summary_print.set_index(['Max_Of', 'Class'])
    print(summary_print.to_latex(float_format=float_format))
    return

def find_recall_val(df:pd.DataFrame, recall_threshold:float, event_col:str='CMIS_MATCH'):
    '''
    05/25/21
    '''
    columns = ['Threshold_Value', 'Number Predicted', 'Actual Number', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    idx = pd.MultiIndex.from_product([['Precision', 'Recall', 'f-1', 'Custom'], ['Positive', 'Negative']], names=['Max_Of', 'Case'])
    summary = pd.DataFrame(index=idx, columns=columns)

    sorted_predictions = df.sort_values(by='prediction')
    for threshold in sorted_predictions['prediction']:
        binned_predictions = (df['prediction'] > threshold)
        current_summary = evaluate_model(binned_predictions=binned_predictions, actuals=df[event_col].astype('bool'))
        if not (current_summary is None):
            '''
            for metric in summary.index.get_level_values('Max_Of').unique():
                if (summary.loc[(metric, 'Positive'), metric] < current_summary.loc['Positive', metric]) or np.isnan(summary.loc[(metric, 'Positive'), metric]):
                    summary.loc[(metric, 'Positive'), 'Number Predicted':'Custom'] = current_summary.loc['Positive']
                    summary.loc[(metric, 'Negative'), 'Number Predicted':'Custom'] = current_summary.loc['Negative']
                    summary.loc[metric, 'Threshold_Value'] = threshold
            if current_summary.loc['Positive', 'Recall'] < recall_threshold:
                summary.loc[('Recall', 'Positive'), 'Number Predicted':'Custom'] = current_summary.loc['Positive']
                summary.loc[('Recall', 'Negative'), 'Number Predicted':'Custom'] = current_summary.loc['Negative']
                summary.loc['Recall', 'Threshold_Value'] = threshold
                return summary
            '''
            if current_summary.loc['Positive', 'Recall'] < recall_threshold:
                return current_summary
    int_cols = ['Number Predicted', 'Actual Number']
    float_cols = ['Threshold_Value', 'False P/N Rate', 'Precision', 'Recall', 'f-1', 'Custom']
    for col in int_cols:
        summary[col] = summary[col].astype('int')
    for col in float_cols:
        summary[col] = summary[col].astype('float')
    return summary