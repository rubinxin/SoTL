# Code to reproduce Fig. 4 in the paper

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# from svr_surrogate import LcSVR

import numpy as np
from scipy import stats
import pickle

# experiment set-up
# n_arch = 50

sum_window_E = 1  # SoLT-E=1
sum_window_E_val = 2

# set-up for computing rank correlation performance of LcSVR
model_name = 'svr'
n_train = 100
svr_interval = 25
include_svr = False
stop_at_max = True
check_minibatch = True
# dataset_list = ['cifar10-valid', 'cifar100', 'ImageNet16-120']

# dataset_list = ['DARTS_8Cells_Cos_CIFAR10_EMA']
# dataset_list = ['DARTS_20Cells_Step_CIFAR10_EMA']
dataset_list = ['DARTS_20Cells_Cos_CIFAR10_EMA']

# for seed in range(5,20):
for seed in [6]:

    for dataset in dataset_list:

        if dataset == 'DARTS_20Cells_CIFAR10':
            max_epochs = 150
            arch_dataset = f'./all_results/cifar10_minibatch_20cells_stepdecay_res'

        elif dataset == 'DARTS_20Cells_Cos_CIFAR10_EMA':
            legend_show = True
            seed = 17
            np.random.seed(seed)
            subset_size = 100
            max_epochs = 150
            arch_dataset = f'./all_results/cifar10_minibatch_20cells_cosine_bs96_ema_res'
            method_list = ['sum-0-train_loss', f'sum-{sum_window_E}-train_loss', 'train_loss_minibatch',
                           'sum-0-val_loss', 'sum-2-val_acc', 'sum-2-ema_val_acc']

        elif dataset == 'DARTS_20Cells_Step_CIFAR10_EMA':
            legend_show = True
            seed = 23
            np.random.seed(seed)
            subset_size = 100
            max_epochs = 150
            arch_dataset = f'./all_results/cifar10_minibatch_20cells_stepdecay_bs128_ema_res'
            method_list = ['sum-0-train_loss', f'sum-{sum_window_E}-train_loss', 'train_loss_minibatch',
                           'sum-0-val_loss',
                           'sum-2-val_acc', 'sum-2-ema_val_acc']

        elif dataset == 'DARTS_8Cells_Cos_CIFAR10_EMA':
            seed = 16
            legend_show = True
            np.random.seed(seed)
            subset_size = 90
            max_epochs = 150
            arch_dataset = f'./all_results/cifar10_minibatch_8cells_cosine_bs128_ema_res'
            method_list = ['sum-0-train_loss', f'sum-{sum_window_E}-train_loss', 'train_loss_minibatch',
                           'sum-0-val_loss',
                           'sum-2-val_acc', 'sum-2-ema_val_acc']

        if check_minibatch:
            figure, axes = plt.subplots(1, 1, figsize=(3, 3))
        else:
            figure, axes = plt.subplots(1, 1, figsize=(3, 3))

        fs = 11

        # ======== load prestored arch data ========
        with open(arch_dataset, 'rb') as outfile:
            res = pickle.load(outfile)
        n_arch = len(res['train_acc'])
        print(f'total_n_arch ={n_arch}')

        rank_correlation_res = {}
        for method_name in method_list:
            print(f'compute rank correlation for {method_name}')
            test_acc_all_arch = res['test_acc']

            if 'sum' in method_name:
                method_name_components = method_name.split('-')
                window_size = int(method_name_components[1])
                metric_name = method_name_components[-1]
                metric_all_arch = res[metric_name]

            else:
                metric_name = method_name
                metric_all_arch = res[metric_name]

            n_arch = len(metric_all_arch)
            if subset_size is not None:
                indices = np.random.choice(range(n_arch), subset_size)
                metric_all_arch = [metric_all_arch[i] for i in indices]
                test_acc_all_arch = [test_acc_all_arch[i] for i in indices]
                n_arch = subset_size
            else:
                indices = range(n_arch)

            test_acc_all_arch_array = np.vstack(test_acc_all_arch)

            if 'sum' in method_name:
                sum_metric_all_arch = []
                for i in range(n_arch):
                    metric_one_arch = metric_all_arch[i]
                    if window_size > 0:
                        so_metric = [np.sum(metric_one_arch[se - window_size:se]) for se in range(window_size, max_epochs)]
                    else:
                        so_metric = [np.sum(metric_one_arch[:se]) for se in range(max_epochs)]
                    sum_metric_all_arch.append(so_metric)

                sum_metric_all_arch_array = np.vstack(sum_metric_all_arch)

                rank_correlation_metric = []
                for j in range(sum_metric_all_arch_array.shape[1]):
                    if 'loss' in metric_name:
                        sum_metric_estimator =  - sum_metric_all_arch_array[:, j]
                    else:
                        sum_metric_estimator =  sum_metric_all_arch_array[:, j]
                    rank_coeff, _ = stats.spearmanr(test_acc_all_arch_array, sum_metric_estimator)
                    rank_correlation_metric.append(rank_coeff)

            else:
                metric_all_arch_array = np.vstack(metric_all_arch)
                rank_correlation_metric = []
                for j in range(metric_all_arch_array.shape[1]):
                    if 'loss' in metric_name:
                        metric_estimator =  - metric_all_arch_array[:, j]
                    else:
                        metric_estimator =  metric_all_arch_array[:, j]
                    rank_coeff, _ = stats.spearmanr(test_acc_all_arch_array, metric_estimator)
                    rank_correlation_metric.append(rank_coeff)

            if 'minibatch' in method_name:
                steps_per_epoch = res['steps_per_epoch']

                n_steps = len(rank_correlation_metric)
                rank_correlation_metric_mean = [
                    np.mean(rank_correlation_metric[k * steps_per_epoch: (k+1) * steps_per_epoch])
                    for k in range(max_epochs)]

                rank_correlation_metric = rank_correlation_metric_mean

            rank_correlation_res[method_name] = rank_correlation_metric


        dic_for_plot = {
        'SoTL-E': [range(sum_window_E, int(sum_window_E + len(rank_correlation_res[f'sum-{sum_window_E}-train_loss']))),
                     rank_correlation_res[f'sum-{sum_window_E}-train_loss'], f'r-'],
        # 'SoTL': [range(1, len(rank_correlation_res['sum-0-train_loss'])), rank_correlation_res['sum-0-train_loss'][1:], 'b-'],
        # 'TLmini': [range(1, len(rank_correlation_res['train_loss_minibatch'])), rank_correlation_res['train_loss_minibatch'][1:], 'k-'],
        # 'SoVL': [range(1, len(rank_correlation_res['sum-0-val_loss'])), rank_correlation_res['sum-0-val_loss'][1:], 'm-'],
        'VAccES': [range(sum_window_E_val, len(rank_correlation_res['sum-2-val_acc'])), rank_correlation_res['sum-2-val_acc'], 'g-'],
        # 'VAccES(EMA)': [range(sum_window_E_val, len(rank_correlation_res['sum-2-ema_val_acc'])), rank_correlation_res['sum-2-ema_val_acc'], 'g--']
        }


        # === plot the rank correlation performance ========
        SoTL_E_max = np.max(np.nan_to_num(rank_correlation_res[f'sum-{sum_window_E}-train_loss']))
        for item in dic_for_plot.items():
            label = item[0]
            content = item[1]
            x_range, rank_corr, fmt = content

            if stop_at_max:
                rank_corr_no_nan = np.nan_to_num(rank_corr)
                max_rank_corr = np.max(rank_corr_no_nan)
                if max_rank_corr <= SoTL_E_max:
                    max_idx = np.where(rank_corr_no_nan == max_rank_corr)[0][0]
                else:
                    max_idx = np.where(rank_corr_no_nan >= SoTL_E_max)[0][0]
                    max_rank_corr = SoTL_E_max

                try:
                    axes.plot(x_range[0: max_idx], rank_corr[0: max_idx], fmt, label=label)
                except:
                    axes.plot(x_range[0: max_idx], rank_corr[0: max_idx-1], fmt, label=label)

                after_max_x_range = x_range[max_idx::10]
                if label == 'SoTL-E':
                    axes.plot(after_max_x_range, [max_rank_corr]*len(after_max_x_range), f'{fmt}-')
                axes.axvspan(100, max_epochs, color='k', alpha=0.1, lw=0, zorder=10)
            else:
                try:
                    axes.plot(x_range[:len(rank_corr)], rank_corr, fmt, label=label)
                except:
                    print('hold')

        # axes.set_xlim([4, 200])
        if legend_show:
            axes.legend(prop={'size': fs-1}, loc="lower right").set_zorder(12)
        axes.set_title(f'{dataset}')
        axes.set_xlim([sum_window_E_val, max_epochs])
        axes.set_xscale('log')
        if dataset == 'DARTS_20Cells_Cos_CIFAR10_EMA':
            axes.set_ylim([0.2, 1])

        axes.set_xlabel('Training Epoch T', fontsize=fs)
        axes.set_ylabel('Rank Correlation', fontsize=fs)
        data_name = arch_dataset.split('/')[-1]
        fig_name = f'./figs/{data_name}_minibatch{check_minibatch}_for{n_arch}archs_on{dataset}_{max_epochs}_{seed}.pdf'
        plt.savefig(fig_name, bbox_inches='tight')
