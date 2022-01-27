import pickle
from time import time
from preprocessing import preprocess
from log_fit import log_fit
from plot_roc import plot_roc
from helper_methods import *

def main():
    time0 = time()
    datapath = "../Data/"
    preprocessed = preprocess(datapath = datapath)
    output = log_fit(preprocessed = preprocessed)
    del preprocessed
    print_dict(output['Data_Retention_Stats'])

    print("Total Time:")
    print(calc_time_from_sec(seconds = time() - time0))

    # plot_roc(output = output)

    # Save Output
    filename = "output.pickle"
    outfile = open(datapath + filename, 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    return

if __name__ == "__main__":
    main()