import numpy as np
from scipy.stats import multivariate_normal

from utils import antiVectorize
from random import randint

# --------------------------------------------------------------
# SHAPE: (n_subjects, n_rois, n_rois)
# --------------------------------------------------------------


# Antivectorize given vector

def antiVectorize(vec, m):
    M = np.zeros((m,m))
    M[np.tril_indices(m,k=-1)] = vec
    M= M.transpose()
    M[np.tril_indices(m,k=-1)] = vec
    return M


# Data simulation function, using multivariate normal. This method simulates the most realistic dataset we obtained so far.

def multivariate_simulate(n_samples=200, data_file):
    # Note that changing the node count is not provided right now, since we use correlation matrix
    # and the mean values of connectivities from real data and it is for 35 nodes.
    # Import all required statistical information.

    adj = []

    allstats = np.load("./stats/REALDATA_" + data_file + "_AVGMEANS.npy",
                       allow_pickle=True)  # Connectivity mean values of LH. You can also try with RH.
    allcorrs = np.load("./stats/REALDATA_" + data_file + "_AVGCORRS.npy",
                       allow_pickle=True)  # Correlation matrix in LH. You can also try with RH.

    # Note that we randomly assign a new random state to ensure it will generate a different dataset at each run.
    # Generate data with the correlations and mean values
    connectomic_means = allstats[0, 0]
    data = multivariate_normal.rvs(connectomic_means, allcorrs[0, 0], n_samples, random_state=randint(1,9999))

    for idx, sample in enumerate(data):
        # Create adjacency matrix.
        matrix = antiVectorize(sample, 35)
        adj.append(matrix)

    alldata = np.array(adj)
    alldata = np.absolute(alldata)
    print(alldata.shape)
    return alldata


def prepare_data(new_data=False, n_samples=200, data_file="1"):
    try:
        if new_data:
            samples = multivariate_simulate(n_samples,n_times)
            np.save('./multivariate_simulation_data_' + data + '.npy', samples)
        else:
            samples = np.load('./multivariate_simulation_data_' + data + '.npy')
    except:
        samples = multivariate_simulate(n_samples,n_times)
        np.save('./multivariate_simulation_data_' + data + '.npy', samples)
    return samples


if __name__ == "__main__":
    X = prepare_data(new_data=True, n_samples=120, data_file="1")