from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

min_metrics = ['jensen', 'kullback_f0_f', 'kullback_f_f0']
path_scores = 'out/scores/'
path_save = 'out/plots/'

# =======================================================
#   Useful functions
# =======================================================


def verify(dataframe, message):
    if dataframe.empty:
        raise ValueError('empty data frame: ' + message)


def set_datasetname(dataset):

    return dataset.replace('_', '-')


def set_metricname(metric):
    if metric == 'auc_anomaly':
        name = 'AUC'
    elif metric == 'jensen':
        name = '$D_{{JS}}$'
    elif metric == 'kullback_f0_f':
        name = '$D_{{KL}}$'
    elif metric == 'kullback_f_f0':
        name = '$D_{{KL}}$'
    else:
        name = metric
    return name


def set_algoname(algo):
    if algo == 'kde':
        return 'KDE'
    elif algo == 'spkde':
        return 'SPKDE'
    elif algo == 'rkde':
        return 'RKDE'
    else:
        return algo


# =======================================================
#   Parameters
# =======================================================
score_file = 'scores_D_2_uniform.csv'

# Which metric ?
# metric = 'jensen'
# metric = 'kullback_f0_f'
# metric = 'kullback_f_f0'
metric = 'auc_anomaly'

SAVE_PLOT = 1
SHOW = 1
LEGEND = 1

# Which dataset ?
datasets = [
    # 'banana',
    # 'titanic',
    # 'german',
    # 'sk-breast-cancer',
    # 'iris',
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
    # 'D_1_uniform',
    # 'D_1_reg_gaussian',
    # 'D_1_thin_gaussian',
    # 'D_1_adversarial',
    'D_2_uniform'
]

# Which methods to plot ?
algos = [
    'kde',
    'rkde',
    'spkde',
    # 'mom-kde'
]

# Plot params
#rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
FIGSIZE = (5, 4)
TINY_SIZE = 8
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=TINY_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=TINY_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
HSPACE = None
# TOP = 0.95
TOP = None
BOTTOM = 0.17
# BOTTOM = None
LEFT = 0.19
# LEFT = None


# =======================================================
#   Processing
# =======================================================
#outlierprop_range = [0.001, 0.02, 0.05,0.07, 0.09, 0.11, 0.13, 0.16, 0.18, 0.2]
outlierprop_range = [0.001, 0.07, 0.09, 0.13, 0.16, 0.2]

x_plot = outlierprop_range

scores = pd.read_csv(path_scores + score_file)

scores_arr_mean = np.zeros((len(algos), len(x_plot)))
scores_arr_std = np.zeros((len(algos), len(x_plot)))

for i_dataset, dataset in enumerate(datasets):
    print('\nDataset: ', dataset)
    # select dataset
    scores_select = scores[scores.dataset == dataset]
    verify(scores_select, 'scores_select, dataset')

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.subplots_adjust(hspace=HSPACE, top=TOP, bottom=BOTTOM, left=LEFT)
    # Setup scores
    for i_algo, algo in enumerate(algos):
        print('Algo', algo)
        tmp_algo = scores_select[scores_select.algo == algo]
        verify(tmp_algo, 'algo')
        for i_outlierprop, outlierprop in enumerate(outlierprop_range):
            tmp_prop = tmp_algo[tmp_algo.outlier_prop == outlierprop]
            verify(tmp_prop, 'epsilon')

            tmp = tmp_prop[metric]
            verify(tmp, 'metric')
            if i_outlierprop == 0:
                print('n exp', tmp.shape)
            score_mean = np.mean(tmp)
            score_std = np.std(tmp)
            scores_arr_mean[i_algo, i_outlierprop] = score_mean
            scores_arr_std[i_algo, i_outlierprop] = score_std

    # Plots
    for i_algo, algo in enumerate(algos):
        if algo == 'kde':
            marker = 'o'
            ls = ''
        elif algo == 'spkde':
            marker = ''
            ls = '--'
        else:
            marker = ''
            ls = '-'
        algo_name = set_algoname(algo)
        ax.plot(x_plot,
                scores_arr_mean[i_algo, :],
                label=algo_name,
                linestyle=ls,
                marker=marker)
        ax.grid(alpha=.3)
        ax.fill_between(x_plot,
                        scores_arr_mean[i_algo, :] - scores_arr_std[i_algo, :],
                        scores_arr_mean[i_algo, :] + scores_arr_std[i_algo, :],
                        alpha=0.2)
        metric_name = set_metricname(metric)
        ax.set_ylabel(metric_name)
        #ax.set_xlabel("$|\mathcal{O}| / n$")
        ax.set_xlabel("Outliers")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.set_title(set_datasetname(dataset))

    if i_dataset == 0:
        if LEGEND:
            ax.legend()

    if SAVE_PLOT:
        name_file_0 = score_file.replace('.csv', '')
        fig.savefig(path_save + name_file_0 + '.png')

if SHOW:
    plt.show()
