import pandas as pd
from log_helper_methods import get_metrics

data = {
    'SPA_PER_ID': [0, 1, 2, 3],
    'CMIS_MATCH': [True, False, True, False],
    'prediction': [0.2, 0.9, 0.4, 0.25]
}
df = pd.DataFrame(data=data)

print(get_metrics(df, y_true='CMIS_MATCH', y_pred='prediction'))