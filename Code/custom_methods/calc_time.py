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