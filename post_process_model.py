import scipy as sp
import numpy as np
import pandas as pd
import sklearn as skl
import time
import re
import pickle
import matplotlib.pyplot as plt

import biom
from mpl_toolkits.basemap import Basemap
from sklearn.decomposition import LatentDirichletAllocation


def plot_communities(lons, lats, colorvals, sizevals, cmap_name='plasma',
                     markerscale=15, res='c', title = ''):
    '''Plots communities onto map of the US'''
    plt.figure(figsize=(14, 7))
    map = Basemap(width=6000000,height=4000000,projection='lcc',
                  resolution=res,lat_0=40,lon_0=-98.)
    map.drawmapboundary(fill_color= '#DAF8F6')
    map.fillcontinents(color='white',lake_color= '#DAF8F6')
    map.drawcoastlines()
    map.drawcountries()
    map.drawstates()

    if isinstance(colorvals, str):
        colors = [colorvals for p in lons]
    else:
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(p) for p in colorvals]
    if isinstance(sizevals, str):
        sizes = [sizevals for p in lons]
    else:
        sizes = [markerscale*p for p in sizevals]
    for lon, lat, s, c in zip(lons, lats, sizes, colors):
        map.plot(lon, lat, marker='o', color=c, markersize=s, latlon=True)
    plt.title(title)
    plt.show()

def output_community_characteristics(model, home_data, biom_data, plots=False):
    '''Return DataFrame of the distribution of communities for each location with geoclimate data'''
    #: Extract document topic distribution.
    model_doc_topic = model.transform(biom_data.matrix_data.transpose().astype('int'))
    print(model_doc_topic.shape)
    n_com = len(model_doc_topic.transpose())
    #: Extract integers from location IDs.
    r = re.compile(r'^(\d+)\.O$')
    int_model_locations = [int(r.match(loc).group(1)) for loc in biom_data.ids('sample')]
    #: Create dataframe of community and location data.
    col_names = ['Community {0}'.format(i) for i in range(n_com)]
    comm_df = pd.DataFrame(model_doc_topic, index=int_model_locations, columns=col_names)
    #: Create list of home data rows in order.
    ordered_geo_dfs = [home_data[home_data['ID'] == loc] for loc in int_model_locations]
    for col in ordered_geo_dfs[0]:
        comm_df[col] = [df[col].values[0] for df in ordered_geo_dfs]
    if plots:
        for i in range(n_com):
            plot_communities(comm_df['Longitude'], comm_df['Latitude'],\
             'r', model_doc_topic[:,i], res='c', title='Community {0}'.format(i))
    return comm_df

def weighted_mean_std(info_data, doc_topic_weight):
    '''Return weighted mean and weighted std as tuple for all communities'''
    com_mean = [np.average(info_data, None, t) for t in doc_topic_weight.transpose()]
    com_var = [np.average((info_data-avg)**2, weights=t)
               for avg, t in zip(com_mean, doc_topic_weight.transpose())]
    com_std = np.sqrt(com_var)
    return (com_mean, com_std)

def plot_wm_wstd(com_mean, com_std, y_label, n):
    '''Plot mean with std as error bars'''
    plt.errorbar(x=list(range(n)), y=com_mean, yerr=com_std, fmt='ro')
    plt.xlabel('Community')
    plt.ylabel(y_label)
    plt.title('Weighted mean and std by community')
    plt.show()

def calc_weights(comm_df, model, randomize=False, \
                 charactaristics=['Latitude', 'Longitude', 'Elevation', \
                                  'MeanAnnual Temperature', 'MeanAnnualPrecipitation'], \
                 titles=['Latitude', 'Longitude', 'Elevation', \
                         'Mean annual temperature', 'Mean annual precipitation'],
                 plots=True):
    '''Return DataFrame of weighted mean and std of each characteristic'''
    #: Extract document topic distribution.
    n_comm = len(model.components_)
    col_names = ['Community {0}'.format(i) for i in range(n_comm)]
    model_doc_topic = np.array(comm_df[col_names])
    #: Create DataFrame of weighted mean and std for each community for each geoclimate parameter
    weighted_parameters = pd.DataFrame()
    for i in range(len(charactaristics)):
        weighted_parameters[charactaristics[i]+' Mean'], weighted_parameters[charactaristics[i]+' Std'] \
                = weighted_mean_std(comm_df[charactaristics[i]], model_doc_topic)
        if plots:
            plot_wm_wstd(weighted_parameters[charactaristics[i]+' Mean'], \
            weighted_parameters[charactaristics[i]+' Std'], \
            titles[i], n=n_comm)

    return weighted_parameters
