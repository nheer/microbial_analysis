import biom
import scipy as sp
import numpy as np
import sklearn as skl
import lda
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
# Run LDA on sparse biom data
def run_multiple_LDA(biom_data, file_name, n_com_list):
    '''Return list of LDA models with number of communities specified in n_com_list

    Extract sparse matrix from biom-format. Run scikit-learn LDA for each number of communities specified.
    Calculate final perplexity of training data and time to run.
    '''
    models = []
    SampleX = biom_data.matrix_data.transpose().astype('int')
    f = open(file_name, 'wb')
    for i in n_com_list:
        starttime = time.time()
        model = LatentDirichletAllocation(n_components=i, learning_method='batch', max_iter=100, evaluate_every=10, max_doc_update_iter=100)
        model.fit(SampleX)
        print('perplexity', model.perplexity(SampleX))
        endtime = time.time()
        print(endtime - starttime)
        pickle.dump(model, f)
        models.append(model)
    return models

def perplexity_plot(model_list, test_data):
    '''Calculate and plot perplexity for list of models using test data.

    Takes list of scikit-learn LDA models followed by withheld data. 
    '''
    test_data = test_data.matrix_data.transpose().astype('int')
    model_topics = [model.n_components for model in model_list]
    perplexities = [model.perplexity(test_data) for model in model_list]
    plt.scatter(model_topics, perplexities)
    plt.xlabel('Number of latent communities')
    plt.ylabel('Perplexity')
    plt.title('Perplexity as a function of the number of latent communities')
    plt.show()
