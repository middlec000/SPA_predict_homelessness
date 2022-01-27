import numpy as np
import pickle
import matplotlib.pyplot as plt
from helper_methods import *

def main():
    datapath = '../Data/'
    filename = 'output0.pickle'

    infile = open(datapath + filename, 'rb')
    model_output = pickle.load(infile)
    infile.close()

    print(filename + ' Features')
    print_list(model_output['Features'])

    plot_roc(output = model_output)
    return


def plot_roc(output):
    summary = output['Performance']

    plt.rcParams.update({
        'figure.figsize':(11,7), 
        'figure.dpi':120
    })
    linewidth = 2
    color = 'black'
    alpha=0.2
    fontsize = 14

    plt.plot(summary['fpr'], summary['tpr'], color='black', label='log', linestyle='-', linewidth=linewidth)

    # Refernce line
    plt.plot([1, 0], [1, 0], color='black', label='y=x', linestyle='dashdot', linewidth=linewidth)
    # Grid
    ticks = np.arange(0, 1.1, 0.1)
    plt.hlines(y=ticks, xmin=0, xmax=1, colors=color, alpha=alpha)
    plt.vlines(x=ticks, ymin=0, ymax=1, colors=color, alpha=alpha)
    # Labels
    plt.xlabel('False Positive Rate ', fontsize=fontsize)
    plt.xticks(ticks, fontsize=fontsize-2)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.yticks(ticks, fontsize=fontsize-2)
    plt.legend(fontsize=fontsize)
    plt.savefig(fname='results_images/ROC.png', bbox_inches='tight')
    plt.show()
    return

if __name__ == "__main__":
    main()