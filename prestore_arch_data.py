import os
import pickle

import numpy as np
from nas_201_api import NASBench201API as API

# Creating an API instance from the NAS-Bench-201 dataset
api = API('./NAS-Bench-201-v1_1-096897.pth')
# randomly sample 5000 archs from a total of 15624 archs
n_arch = 5000
n_arch_max = 15624
nepochs = 200
seed = 888

# all three image datasets
dataset_list = ['cifar10-valid', 'ImageNet16-120', 'cifar100']

for dataset in dataset_list:

    arch_dataset = f'./{dataset}s{seed}_valid_arch_info{n_arch}'

    if not os.path.exists(arch_dataset):

        print(f'{dataset},{seed},generate data')
        valid_acc_seed_all_arch = []
        test_acc_seed_all_arch = []
        train_acc_seed_all_arch = []
        valid_loss_seed_all_arch = []
        test_loss_seed_all_arch = []
        train_loss_seed_all_arch = []
        arch_seed_all_arch = []
        costs_seed_all_arch = []

        # architecture meta-features and training hyperparamters for SVR regressor
        AP_seed_all_arch = []
        HP_seed_all_arch = []

        j = 0
        while j <= 15624:
            # get the detailed information
            results = api.query_by_index(j, dataset)
            arch = api.arch(index=j)

            try:
                print(f'{dataset},{seed},arch_id={j}')

                # get train and validation loss/accuracy and train time
                train_loss_i = []
                val_loss_i = []
                val_acc_i = []
                train_acc_i = []
                train_time_i = []

                # get epoch-wise information
                for e in range(nepochs):
                    results_e = api.get_more_info(j, dataset, e, False, is_random=seed)
                    train_loss_i_e = results_e['train-loss']
                    train_acc_i_e = results_e['train-accuracy']
                    train_time_i_e = results_e['train-all-time']

                    # val accuracy
                    if dataset == 'cifar10-valid':
                        val_acc_i_e = results_e['valid-accuracy']
                        val_loss_i_e = results_e['valid-loss']
                    else:
                        val_acc_i_e = results_e['valtest-accuracy']
                        val_loss_i_e = results_e['valtest-loss']

                    train_loss_i.append(train_loss_i_e)
                    train_acc_i.append(train_acc_i_e)
                    train_time_i.append(train_time_i_e)
                    val_loss_i.append(val_loss_i_e)
                    val_acc_i.append(val_acc_i_e)

                # get final test loss/accuracy
                results_final = api.get_more_info(j, dataset, None, False, is_random=seed)
                test_loss_i = results_final['test-loss']
                test_acc_i = results_final['test-accuracy']

                # collect meta-features and cost information
                cost_info = api.get_cost_info(j, dataset)
                cost_results_i = {'flops': cost_info['flops'], 'params': cost_info['params'],
                                  'latency': cost_info['latency'], 'train_time': train_time_i}

                # hyper_i: lr, momentum, wd, bs,
                hyper_i = [0.1, 0.9, 0.0005, 256]
                # arch_param_i: params, flops, latency
                arch_param_i = [cost_info['params'], cost_info['flops'], cost_info['latency']]

                # store results for valid architectures
                if test_acc_i >= 10:
                    arch_seed_all_arch.append(arch)
                    costs_seed_all_arch.append(cost_results_i)
                    train_loss_seed_all_arch.append(train_loss_i)
                    train_acc_seed_all_arch.append(train_acc_i)
                    valid_loss_seed_all_arch.append(val_loss_i)
                    test_loss_seed_all_arch.append(test_loss_i)
                    valid_acc_seed_all_arch.append(val_acc_i)
                    test_acc_seed_all_arch.append(test_acc_i)
                    HP_seed_all_arch.append(hyper_i)
                    AP_seed_all_arch.append(arch_param_i)

                j += 1

            except:
                j += 1
                print('missing arch info')
                continue

        arch_indices_to_include = np.random.choice(range(len(arch_seed_all_arch)), n_arch, replace=False)

        res = {'arch': [arch_seed_all_arch[i] for i in arch_indices_to_include],
               'meta_cost_info': [costs_seed_all_arch[i] for i in arch_indices_to_include],
               'train_loss': [train_loss_seed_all_arch[i] for i in arch_indices_to_include],
               'train_acc': [train_acc_seed_all_arch[i] for i in arch_indices_to_include],
               'val_loss': [valid_loss_seed_all_arch[i] for i in arch_indices_to_include],
               'val_acc': [valid_acc_seed_all_arch[i] for i in arch_indices_to_include],
               'test_loss': [test_loss_seed_all_arch[i] for i in arch_indices_to_include],
               'test_acc': [test_acc_seed_all_arch[i] for i in arch_indices_to_include],
               'AP': [AP_seed_all_arch[i] for i in arch_indices_to_include],
               'HP': [HP_seed_all_arch[i] for i in arch_indices_to_include]}

        with open(arch_dataset, 'wb') as outfile:
            pickle.dump(res, outfile)
