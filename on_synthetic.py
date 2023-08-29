from libs import kde_lib
from libs import data
from libs.exp_lib import Density_model
import numpy as np
from sklearn.datasets import make_classification


# =======================================================
#   Parameters
# =======================================================
algos = [
    'kde',
    'rkde',
    'spkde',
]

# Synth params
dim = 1
n_samples = 500
outlier_scheme = 'uniform'
#outlier_scheme = 'reg_gaussian'
#outlier_scheme = 'thin_gaussian'
# outlier_scheme = 'adversarial'

#dataset = ('D_' + str(dim) + '_' + outlier_scheme)
X0, y0 = make_classification(
    500, n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1
)
dataset = "synthetic_data"

#outlierprop_range = [0.001, 0.02, 0.05,0.07, 0.09, 0.11, 0.13, 0.16, 0.18, 0.2]
outlierprop_range = [0.001, 0.07, 0.09, 0.13, 0.16, 0.2]
grid_points = 500
kernel = 'gaussian'

n_exp = 3

WRITE_SCORE = 1
scores_file = "out/scores/scores_{}.csv".format(dataset)

# =======================================================
#   Processing
# =======================================================


def generate_outliers(X, y, outlier_proportion=.1):

    # Define the proportion of outliers to add
    outlier_proportion = outlier_proportion
    # Calculate the number of outliers to add
    num_outliers = int(outlier_proportion * len(X))
    # Generate random outlier points within the range of the dataset
    outliers_X = np.random.rand(
        num_outliers, 2) * (np.max(X, axis=0) - np.min(X, axis=0)) + np.min(X, axis=0)
    # Assign a class label to outliers
    outliers_y = np.array([1] * num_outliers)

    # Concatenate outliers with the original dataset
    X = np.vstack((X, outliers_X))
    y = np.concatenate((y, outliers_y))
    return X, y


for i_exp in range(n_exp):
    print('EXP: {} / {}'.format(i_exp + 1, n_exp))
    for i_outlierprop, outlier_prop in enumerate(outlierprop_range):
        print('\nOutlier prop: {} ({} / {})'.format(outlier_prop,
              i_outlierprop + 1, len(outlierprop_range)))
        """ epsilon = outlier_prop / (1 - outlier_prop)
        X, y = data.generate_situations(n_samples,
                                        outlier_prop,
                                        outlier_scheme=outlier_scheme,
                                        dim=dim) """
        X, y = generate_outliers(X0, y0, outlier_prop)
        #  Grid on which to evaluate densities
        grid, X_plot = data.make_grid(X, grid_points)
        n_outliers = np.sum(y == 0)
        #  Find bandwidth
        X_inlier = X[y == 1]

        h_hho = kde_lib.hho_bandwith_selection(X_inlier, X_plot)
        #h = h_hho
        h_cvgrid, _, _ = kde_lib.bandwidth_cvgrid(X_inlier)
        #h = h_cvgrid
        true_dens = data.true_density_situations(X_plot)

        #  Processing all algos
        for algo in algos:
            print('\nAlgo: ', algo)
            h = (lambda: h_cvgrid, lambda: h_hho)[algo == "rkde"]()
            kde_model = Density_model(
                algo, dataset, outlier_prop, kernel, h)

            kde_model.fit(X, X_plot, grid)
            kde_model.compute_score(true_dens)

            kde_model.estimate_density(X)
            kde_model.compute_anomaly_roc(y)
            """ if epsilon != 0:
                # density on observation
                kde_model.estimate_density(X)
                kde_model.compute_anomaly_roc(y) """
            if WRITE_SCORE:
                kde_model.write_score(scores_file)
