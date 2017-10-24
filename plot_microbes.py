import matplotlib.pyplot as plt
import numpy as np

def log_hist_plot(data, title='', xlabel='', ylabel=''):
    '''Plot log-scale and linear-scale data with 50 bins.

    Used for plotting number of counts per location and per taxa.
    '''
    data = data[data > 1e-7].transpose()
    plt.hist(data, bins=np.logspace(np.log10(data.min()), np.log10(data.max()), 50))
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.hist(data, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
