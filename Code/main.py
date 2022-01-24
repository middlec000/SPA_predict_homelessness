from time import time
from preprocessing import preprocess
from log_fit import log_fit
from plot_roc import plot_roc
from helper_methods import *

def main():
    time0 = time()
    preprocessed = preprocess()
    output = log_fit(preprocessed = preprocessed)
    del preprocessed
    print_dict(output['Data_Retention_Stats'])
    plot_roc(output = output)
    calc_time_from_sec(seconds = time() - time0)
    return

if __name__ == "__main__":
    main()