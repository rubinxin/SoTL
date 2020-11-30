# Code to reproduce Fig. 4 in the paper

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from svr_surrogate import LcSVR

import numpy as np
from scipy import stats
import pickle

# experiment set-up
max_epochs = 200
n_arch = 500
seed = 888
sum_window_E = 1  # SoLT-E=1

# set-up for computing rank correlation performance of LcSVR
model_name = 'svr'
n_train = 200
svr_interval = 25
include_svr = True

# dataset_list = ['cifar10-valid', 'cifar100', 'ImageNet16-120']
dataset_list = ['ImageNet16-120']


for dataset in dataset_list:

    figure, axes = plt.subplots(1, 1, figsize=(3, 4))
    fs = 11
    
    # ======== load prestored arch data ========
    arch_dataset = f'./{dataset}s{seed}_valid_arch_info{n_arch}'
    with open(arch_dataset, 'rb') as outfile:
        res = pickle.load(outfile)
    
    
    train_loss_all_arch = res['train_loss']
    valid_loss_all_arch = res['val_loss']
    valid_acc_all_arch = res['val_acc']
    test_loss_all_arch = res['test_loss']
    test_acc_all_arch = res['test_acc']
    AP_all_arch = res['AP']
    HP_all_arch = res['HP']


    # ===== compute estimators: SoTL, SoTL-E, SoVL, SoVL-E, SoVAcc-E for each arch ========
    SoTL_all_arch = []
    SoTL_E_all_arch = []

    SoVL_all_arch = []
    SoVL_E_all_arch = []
    SoVAcc_E_all_arch = []

    # loop through all the archs
    for j in range(len(train_loss_all_arch)):

        # compute SoTL and SoTL-E
        train_loss = train_loss_all_arch[j]
        SoTL = [np.sum(train_loss[:se]) for se in range(max_epochs)]
        SoTL_E = [np.sum(train_loss[se - sum_window_E:se]) for se in range(sum_window_E, max_epochs)]

        SoTL_all_arch.append(SoTL)
        SoTL_E_all_arch.append(SoTL_E)


        # compute SoVL, SoVL-E, SoVAcc-E
        valid_loss = valid_loss_all_arch[j]
        SoVL = [np.sum(valid_loss[:se]) for se in range(max_epochs)]
        SoVL_E = [np.sum(valid_loss[se - sum_window_E:se]) for se in range(sum_window_E, max_epochs)]
        valid_acc = valid_acc_all_arch[j]
        SoVAcc_E = [np.sum(valid_acc[se - sum_window_E:se]) for se in range(sum_window_E, max_epochs)]

        SoVL_all_arch.append(SoVL)
        SoVL_E_all_arch.append(SoVL_E)
        SoVAcc_E_all_arch.append(SoVAcc_E)

    # training-related estimators: SoTL, SoTL-E
    SoTL_all_arch_array = np.vstack(SoTL_all_arch)
    SoTL_E_all_arch_array = np.vstack(SoTL_E_all_arch)

    # validation-related estimators: SoVL, SoVL-E, Valid Acc, SoVAcc-E
    SoVL_all_arch_array = np.vstack(SoVL_all_arch)
    SoVL_E_all_arch_array = np.vstack(SoVL_E_all_arch)

    valid_acc_all_arch_array = np.vstack(valid_acc_all_arch)
    SoVAcc_E_all_arch_array = np.vstack(SoVAcc_E_all_arch)

    # final test accuracies and loss
    test_acc_all_arch_array = np.vstack(test_acc_all_arch)
    test_loss_all_arch_array = np.vstack(test_loss_all_arch)

    # === compute rank correlation between the final test accuracy and estimators over training epochs ===

    # rank correlation bw final test accuracy and - SoTL
    rank_correlation_SoTL = []
    for j1 in range(SoTL_all_arch_array.shape[1]):
        rank_coeff1, _ = stats.spearmanr(test_acc_all_arch_array, - SoTL_all_arch_array[:,j1])
        rank_correlation_SoTL.append(rank_coeff1)

    # rank correlation bw final test accuracy and - SoTL-E
    rank_correlation_SoTL_E = []
    for j2 in range(SoTL_E_all_arch_array.shape[1]):
        rank_coeff2, _ = stats.spearmanr(test_acc_all_arch_array, - SoTL_E_all_arch_array[:,j2])
        rank_correlation_SoTL_E.append(rank_coeff2)

    # rank correlation bw final test accuracy and Val Acc
    rank_correlation_val_acc = []
    for j3 in range(valid_acc_all_arch_array.shape[1]):
        rank_coeff3, _ = stats.spearmanr(test_acc_all_arch_array, valid_acc_all_arch_array[:, j3])
        rank_correlation_val_acc.append(rank_coeff3)

    # rank correlation bw final test accuracy and - SoVL
    rank_correlation_SoVL = []
    for j4 in range(SoVL_all_arch_array.shape[1]):
        rank_coeff4, _ = stats.spearmanr(test_acc_all_arch_array, - SoVL_all_arch_array[:,j4])
        rank_correlation_SoVL.append(rank_coeff4)

    # rank correlation bw final test accuracy and  - SoVL-E
    rank_correlation_SoVL_E = []
    for j5 in range(SoVL_E_all_arch_array.shape[1]):
        rank_coeff5, _ = stats.spearmanr(test_acc_all_arch_array, - SoVL_E_all_arch_array[:,j5])
        rank_correlation_SoVL_E.append(rank_coeff5)

    # rank correlation bw final test accuracy and  - SoVAcc-E
    rank_correlation_SoVAcc_E = []
    for j6 in range(SoVAcc_E_all_arch_array.shape[1]):
        rank_coeff6, _ = stats.spearmanr(test_acc_all_arch_array, - SoVAcc_E_all_arch_array[:,j6])
        rank_correlation_SoVAcc_E.append(rank_coeff6)

    # rank correlation bw final test accuracy and - final test loss
    rank_correlation_test_loss, _ = stats.spearmanr(test_acc_all_arch_array, - test_loss_all_arch_array)

    # rank correlation bw final test accuracy and negative prediction by frequestist regression models
    if include_svr:
        # initialise the LcSVR regression model
        svr_regressor = LcSVR(valid_acc_all_arch, HP_all_arch, AP_all_arch, test_acc_all_arch,
                              all_curve=True, n_train=n_train)
        # rank correlation bw final test accuracy and predicted accuracy by SVR regression models
        rank_correlation_LcSVR = []
        epoch_list = range(25, max_epochs+1, svr_interval)
        for epoch in epoch_list:
            best_hyper, time_taken = svr_regressor.learn_hyper(epoch)
            rank_coeff8 = svr_regressor.extrapolate()
            rank_correlation_LcSVR.append(rank_coeff8)

        dic_for_plot = {
            'SoTL-E': [range(sum_window_E, int(sum_window_E+len(rank_correlation_SoTL_E))),
                        rank_correlation_SoTL_E, f'r-'],
            'SoTL': [range(1, 1+len(rank_correlation_SoTL)), rank_correlation_SoTL[1:], 'b-'],
            'SoVL': [range(1, 1+len(rank_correlation_SoVL)), rank_correlation_SoVL[1:], 'm-'],
            'Val Acc': [range(len(rank_correlation_val_acc)), rank_correlation_val_acc, 'g-'],
            'LcSVR': [epoch_list, rank_correlation_LcSVR, 'c-']
            }
    else:
        dic_for_plot = {
            'SoTL-E': [range(sum_window_E, int(sum_window_E + len(rank_correlation_SoTL_E))),
                       rank_correlation_SoTL_E, f'r-'],
            'SoTL': [range(1, 1 + len(rank_correlation_SoTL)), rank_correlation_SoTL[1:], 'b-'],
            'SoVL': [range(1, 1 + len(rank_correlation_SoVL)), rank_correlation_SoVL[1:], 'm-'],
            'Val Acc': [range(len(rank_correlation_val_acc)), rank_correlation_val_acc, 'g-'],
        }

    # === plot the rank correlation performance ========

    SoTL_E_max = np.max(np.nan_to_num(rank_correlation_SoTL_E))
    for item in dic_for_plot.items():
        label = item[0]
        content = item[1]
        x_range, rank_corr, fmt = content
        
        rank_corr_no_nan = np.nan_to_num(rank_corr)
        max_rank_corr = np.max(rank_corr_no_nan)
        if max_rank_corr <= SoTL_E_max:
            max_idx = np.where(rank_corr_no_nan == max_rank_corr)[0][0]
        else:
            max_idx = np.where(rank_corr_no_nan >= SoTL_E_max)[0][0]
            max_rank_corr = SoTL_E_max

        axes.plot(x_range[0: max_idx], rank_corr[0: max_idx], fmt, label=label)
        after_max_x_range = x_range[max_idx::10]
        if label == 'SoTL-E':
            axes.plot(after_max_x_range, [max_rank_corr]*len(after_max_x_range), f'{fmt}-')
        axes.axvspan(100, max_epochs, color='k', alpha=0.1, lw=0, zorder=10)
        

    axes.set_xlim([4, 200])
    axes.legend(prop={'size': fs-1}, loc="lower right").set_zorder(12)
    axes.set_title(f'{dataset}')
    axes.set_ylim([0.6, 1])
    axes.set_xscale('log')

    axes.set_xlabel('Training Epoch T', fontsize=fs)
    axes.set_ylabel('Rank Correlation', fontsize=fs)
    fig_name = f'./Rank_corr_performance_of_various_estimators_for{n_arch}archs_on{dataset}.pdf'
    plt.savefig(fig_name, bbox_inches='tight')
