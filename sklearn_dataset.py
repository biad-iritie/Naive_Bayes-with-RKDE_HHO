from libs import kde_lib
from libs import data
from libs.exp_lib import Density_model
import numpy as np


# =======================================================
#   Parameters
# =======================================================
algos = [
    'kde',
    'rkde',
    'spkde',
]

datasets = [
    # 'banana',
    # 'titanic',
    # 'german',
    # 'sk-breast-cancer',
    'iris',
    # 'digits_0',
    # 'digits_1',
    # 'digits_2',
    # 'digits_3',
    # 'digits_4',
    # 'digits_5',
    # 'digits_6',
    # 'digits_7',
    # 'digits_8',
    # 'digits_9',
    # 'digits_1_0',
    # 'digits_0_1',
]

n_exp = 3
outlierprop_range = [0.001, 0.02, 0.05,
                     0.07, 0.09, 0.11, 0.13, 0.16, 0.18, 0.2]
kernel = 'gaussian'

WRITE_SCORE = 1
scores_file = "out/scores/scores_{}.csv".format(datasets[0])

# =======================================================
#   Processing
# =======================================================

for i_exp in range(n_exp):
    print('EXP: {} / {}'.format(i_exp + 1, n_exp))
    for dataset in datasets:
        print('Dataset: ', dataset)
        X0, y0 = data.load_data_outlier(dataset)
        for i_outlierprop, outlier_prop in enumerate(outlierprop_range):
            epsilon = outlier_prop / (1 - outlier_prop)
            print('\nOutlier prop: {} ({} / {})'.format(outlier_prop,
                  i_outlierprop + 1, len(outlierprop_range)))
            # balance the inlier / outlier according to epsilon
            if epsilon != 0:
                X, y = data.balance_outlier(X0, y0, e=epsilon)
            else:
                X = X0.copy()
                y = y0.copy()
            n_outliers = np.sum(y == 0)
            # evaluate on observations
            X_plot = X
            # compute true density
            X_inlier = X0[y0 == 1]
            # Find bandwidth
            h_cvgrid, _, _ = kde_lib.bandwidth_cvgrid(X0)
            #h = h_cv
            h_hho = kde_lib.hho_bandwith_selection(X0, X_plot)
            #h = h_hho

            #true_dens = kde_lib.kde(X_inlier, X_plot, h, 'gaussian')
            # set range for k (number blocks) according to outliers
            if epsilon == 0:
                k_range = [1]
            else:
                if epsilon < 1 / 3:
                    k_max = 2 * n_outliers + 1
                else:
                    k_max = X.shape[0] / 2
                k_range = np.linspace(1, k_max, 20).astype(int)
            # Processing all algos
            for algo in algos:
                print('Algo: ', algo)
                h = (lambda: h_cvgrid, lambda: h_hho)[algo == "rkde"]()
                model = Density_model(
                    algo, dataset, outlier_prop, kernel, h)
                model.fit(X, X_plot, grid=None)
                if epsilon != 0:
                    # density on observation
                    # kde_model.estimate_density(X)
                    model.compute_anomaly_roc(y)
                if WRITE_SCORE:
                    model.write_score(scores_file)
