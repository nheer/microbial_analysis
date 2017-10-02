import time
import re
import pickle

import biom
import scipy as sp
import numpy as np
import pandas as pd
import sklearn as skl
import importlib
import matplotlib.pyplot as plt

home_data = pd.read_csv('homes_mapping_comma.csv')

def remove_indoors(values, sampleid, metadata):
    '''Return true if sampleID represents an outdoor location.

    Pass to biom-format.Table.filter.
    '''
    r = re.match(r"^(\d+)\.O$", sampleid)
    if r is None:
        return False
    return int(r.group(1)) in home_data['ID'].values

def process_biom_data(table_name, remove_indoor=True, select_bac=False,
                      remove_bac=False, remove_empty=True):
    '''Remove indoor locations and empty rows/columns from biom-format file .

    Also remove bacteria or mitochondira if option selected.
    Returns biom-format file.
    '''
    # Load in biom-format data
    seqData = biom.load_table(table_name)
    print('Original shape:', seqData.shape)
    # Remove indoor and extranious samples
    if remove_indoors:
        seqData.filter(remove_indoors)
        print('shape with only outdoor locations:', seqData.shape)
    if select_bac:
        seqData.filter(filter_in_bacteria, axis='observation')
    if remove_bac:
        seqData.filter(filter_out_bacteria, axis='observation')
    # Remove Taxa that are not present in outdoor samples
    if remove_empty:
        seqData.remove_empty()
        print('shape with empty removed:', seqData.shape)
    return (seqData)


def filter_out_bacteria(values, sampleid, metadata):
    '''Return true if metadata contain mitochondria or chloroplast.

    Pass to biom-format.Table.filter.
    '''
    meta_search = ' '.join(metadata['taxonomy'][:])
    r_meta = re.search(r"(mitochondria|Chloroplast)", meta_search)
    return r_meta is not None

def filter_in_bacteria(values, sampleid, metadata):
    '''Return true if metadata does not contain mitochondria or chloroplast.

    Pass to biom-format.Table.filter.
    '''
    meta_search = ' '.join(metadata['taxonomy'][:])
    r_meta = re.search(r"(mitochondria|Chloroplast)", meta_search)
    return r_meta is None

def randomize_locations(biom_data, file_name):
    '''Return randomized list of locations and saves list with pickle.

    Input filtered biom_data and name to save file with.
    '''
    random_loc = biom_data.ids('sample')
    np.random.shuffle(random_loc)
    #save list of random locations
    with open(file_name, 'wb') as f:
        pickle.dump(random_loc, f)
    return random_loc

def get_subset(random_loc, data, data_name, division=1000):
    '''Return training and testing datasets in biom-format.

    Input randomized locations, full biom-format dataset, filename for saving.
    Defauts to 1000 test cases with "division" arg. Saves locations associated
    each dataset for retrival.
    '''
    train_data = data.filter(random_loc[:division], inplace=False)
    test_data = data.filter(random_loc[division:], inplace=False)
    # Save the random order of locations
    with open(data_name + '_train,pkl', 'wb') as f:
        pickle.dump(train_data.ids('sample'), f)
    with open(data_name + '_test.pkl', 'wb') as f:
        pickle.dump(test_data.ids('sample'), f)
    return (train_data, test_data)
