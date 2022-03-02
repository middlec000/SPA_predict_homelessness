import pickle
from time import time
from preprocessing import preprocess
from log_fit import log_fit
from plot_roc import plot_roc
from helper_methods import *

def main():
    t0 = time()
    datapath = "../Data/"

    preprocessed = preprocess(datapath = datapath)
    t1 = time()
    print("\nPreprocessing Time:")
    print(calc_time_from_sec(t1-t0))

    output = log_fit(preprocessed = preprocessed)
    t2 = time()
    print("\nFitting Time:")
    print(calc_time_from_sec(t2-t1))
    del preprocessed

    print('\nData Retention Stats:')
    print_dict(output['Data_Retention_Stats'])

    print("Total Time:")
    print(calc_time_from_sec(seconds = time() - t0))

    plot_roc(output = output)

    # Save Output
    filename = "output0.pickle"
    outfile = open(datapath + filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    return

if __name__ == "__main__":
    main()