import numpy as np
from typing import Union

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

def geoid_map(x):
    '''
    05/06/21
    Maps GEOID from Census data to GEOID's in Avista data
    '''
    return x[9:]