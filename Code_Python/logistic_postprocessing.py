import pandas as pd
import pickle
import time
from sklearn.metrics import auc, roc_curve
from custom_methods.calc_time import calc_time_from_sec

startTime = time.time()

datapath = '../Data/'

# Helper method
def get_metrics(df: pd.DataFrame, y_true: str, y_pred: str) -> pd.DataFrame:
    # Overall Counts
    total = len(df)
    p = df[y_true].sum()
    n = total - p
    # True Positive Rate (TPR)/Sensitivity/Recall/Hit Rate
    # False Positive Rate (FPR)/False Alarm Rate
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

# # Load Data
filename = 'log_output.pickle'
infile = open(datapath+filename,'rb')
df = pickle.load(infile)
infile.close()


# ## Extract IDs
df[['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']] = df['PER-PREM-MONTH_ID'].str.split(pat='-', expand=True)

df = df.drop('PER-PREM-MONTH_ID', axis=1)

for col in ['SPA_PER_ID', 'SPA_PREM_ID', 'MONTH']:
    df[col] = df[col].astype('int')


# # Take maximum risk prediction for each person

# ## Take Max Likelihood For Each Person
predictions = df.groupby('SPA_PER_ID')['prediction'].max()
predictions = pd.concat([predictions, df.groupby('SPA_PER_ID').CMIS_MATCH.any()], axis=1, join='inner', ignore_index=False)
predictions = predictions.reset_index()

# Get metrics at each possible threshold
summary = get_metrics(df=predictions, y_true='CMIS_MATCH', y_pred='prediction')

print(summary.head())

# # Save and Time
filename = 'log_summary.pickle'
outfile = open(datapath + filename, 'wb')
pickle.dump(summary, outfile)
outfile.close()

print(calc_time_from_sec(time.time() - startTime))

