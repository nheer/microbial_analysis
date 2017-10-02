import biom
import scipy as sp
import numpy as np
import pandas as pd
import sklearn as skl
import time
import re
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.decomposition import LatentDirichletAllocation

# Extract metadata associated with each taxa and format
def extract_taxa(model, biom_data):
    '''Return DataFrame and matrix of the distributino of OTU in each community'''
    model_topic_word = model.components_/model.components_.sum(axis=1)[:, np.newaxis]
    n_com = len(model_topic_word)
    col_names = ['Community {0}'.format(i) for i in range(n_com)]
    taxonomies = biom_data.metadata(None, axis = 'observation')
    phylla = [' '.join(taxa['taxonomy'][0:5]) for taxa in taxonomies]
    phylla = [p.replace('p__', '').replace('k__', '').replace('c__','').replace('o__','').replace('f__','') for p in phylla]
    # Make DF of the metadata
    df_phylla = pd.DataFrame(model_topic_word.transpose(), index=phylla, columns=col_names)
    # df_phylla = df_phylla.groupby(df_phylla.index).sum()
    df_phylla = df_phylla.rename({'Bacteria': 'Bacteria Unknown', 'Archaea': 'Archaea Unknown'})

    return (df_phylla)

def display_comm_taxa(df_phylla, words=True, graph=False):
    '''Return list of pd.Series containing taxa of top 5 OTUs for each community'''
    top_5 = []
    n_com = len(df_phylla.columns)
    for i in range(n_com):
        temp = df_phylla.sort_values('Community {}'.format(i), ascending=False).head(5)
        top_5.append(temp['Community {}'.format(i)])
    for i in range(n_com):
        if words:
            print(top_5[i])
        if graph:
            top_5[i].plot.bar()
            plt.show()
    return top_5

def comm_distributions(model):
    '''Plot histogram of OTU weights for each community'''
    model_topic_word = model.components_/model.components_.sum(axis=1)[:, np.newaxis]
    for t in range(len(model_topic_word)):
        log_hist_plot(model_topic_word[t,:], title='Community {}'.format(t), xlabel='Weight of a species in a community', ylabel='Number of species')
