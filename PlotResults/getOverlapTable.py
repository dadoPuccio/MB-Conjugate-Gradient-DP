import os
import ast
import argparse

import numpy as np

from plotUtils import *
from plotConfigs import *

# To understand wheter an algorithm works better with or without the usage of overlap we compare the objective function values
# of the two at the time of termination of the first. It means that when one of the two terminates, we check whether the other at that time has
# a better or worse objective function value. 
# Since we sample the true objective at the end of each epoch, the estimated objective value of the second-to-terminate algorithm
# is estimated through interpolation

parser = argparse.ArgumentParser()

parser.add_argument('-dir', '--exp_dir')

args = parser.parse_args()

PROBLEMS = ['mushrooms', 'ijcnn', 'rcv1', 'mnist', 'fashion', 'cifar10']

base_dir = args.exp_dir

custom_filter_convex = {
        "sgd": {'lr': 0.01, 'momentum': 0.9},
        "adam": {"lr": 0.001},
}

custom_filter_non_convex = {
        "sgd": {'lr': 0.001, 'momentum': 0.9}, 
        "adam": {"lr": 0.00001},
}

to_be_analized = {}

for problem in PROBLEMS:

    if problem in ['ijcnn', 'rcv1', 'mushrooms']:
        custom_filter = custom_filter_convex
    elif problem in ['mnist', 'fashion', 'cifar10']:
        custom_filter = custom_filter_non_convex
    else:
        raise ValueError("No such problem")


    for entry in [os.path.join(base_dir, e) for e in sorted(os.listdir(base_dir))]:

        params = read_json(os.path.join(entry, "params.json"))

        if filter_removed(params, custom_filter):
            # print("Removed", params["opt"])
            continue

        if params['dataset'] != problem:
            # print(params['dataset'], problem, params["opt"]['name'])
            continue

        reference_params = params
        
        epoch_stats = read_csv(os.path.join(entry, "epoch_stats.csv"))
        minibatch_stats = read_csv(os.path.join(entry, "minibatch_stats.csv"))

        if 'time' in epoch_stats and 'train_loss' in epoch_stats:
            x = epoch_stats['time']
            y = epoch_stats['train_loss']

        elif 'time' in minibatch_stats and 'train_loss' in minibatch_stats:
            x = minibatch_stats['time']
            y = minibatch_stats['train_loss']

        else:
            raise FileNotFoundError("Unable to read the x_axis / y_axis specified, perhaps wrong axis name in input?")
        
        current_key = str(params['dataset']) + "_" + str(params["batch_size"]) + "_" + str({k: v for k, v in params['opt'].items() if k not in ['max_beta', 'max_eta'] }) + "_" + str(params["overlap_batches"])

        if current_key not in to_be_analized:
            to_be_analized[current_key] = {}

        if params["runs"] in to_be_analized[current_key].keys():
            raise ValueError("There is a mistake in the filtering operation")

        to_be_analized[current_key][params["runs"]] = {'x': x, 'y': y, 'params': params}


meaned_runs = {}

for opt_signature, runs in sorted(to_be_analized.items()):

    print(opt_signature, runs.keys())
    
    max_len = max([len(r['x']) for r in runs.values()])

    all_x = []
    all_y = []
    for r in runs.values():
        all_x.append(np.pad(r['x'], (0, max_len - len(r['x'])), 'linear_ramp', end_values=(r['x'][-1] + (max_len - len(r['x'])) * np.mean([r['x'][i+1] - r['x'][i] for i in range(len(r['x']) - 1)]),)))
        all_y.append(np.pad(r['y'], (0, max_len - len(r['y'])), 'constant'))

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    mean_x = np.mean(all_x, axis=0)

    mean_y = np.mean(all_y, axis=0)
    std_y = np.std(all_y, axis=0)

    min_y = np.amin(all_y, axis=0)
    max_y = np.amax(all_y, axis=0)

    label = r['params']['opt']['name']

    meaned_runs[opt_signature] = {'x': mean_x, 'y': mean_y}


part_opt_signatures = list(set([k[:-5] if k.endswith('_True') else k[:-6] for k in meaned_runs.keys()]))
table_entries = {}

for part_opt_s in part_opt_signatures:

    overlap_false_x = meaned_runs[part_opt_s + "_False"]['x']
    overlap_false_y = meaned_runs[part_opt_s + "_False"]['y']
    
    overlap_true_x = meaned_runs[part_opt_s + "_True"]['x']
    overlap_true_y = meaned_runs[part_opt_s + "_True"]['y']
    
    if (overlap_false_x[-1] <= overlap_true_x[-1]):
        slower_x = overlap_true_x
        slower_y = overlap_true_y
        smaller_last = overlap_false_x[-1] 
        value_quicker = overlap_false_y[-1]
        first_closing = "std"
    else:
        slower_x = overlap_false_x
        slower_y = overlap_false_y
        smaller_last = overlap_true_x[-1] 
        value_quicker = overlap_true_y[-1]
        first_closing = "overlap"
        
    index = -1
    for i, value in enumerate(slower_x):
        if value < smaller_last:
            index = i
        else:
            break
        
    print(smaller_last, slower_x[index], slower_x[index + 1])
    print(value_quicker, slower_y[index], slower_y[index + 1])

    estimated_y = slower_y[index] + (slower_y[index + 1] - slower_y[index]) / (slower_x[index + 1] - slower_x[index]) * (smaller_last - slower_x[index])

    print(estimated_y, "\n")

    if estimated_y < value_quicker:
        if first_closing == 'std':
            best = 'overlap'
        else:
            best = 'std'
    else:
        best = first_closing

    if part_opt_s.split("_")[0] +"_"+ part_opt_s.split("_")[1] not in table_entries:
        table_entries[part_opt_s.split("_")[0] +"_"+ part_opt_s.split("_")[1]] = {}
    
    d = ast.literal_eval("{" + part_opt_s.split("{")[1].split("}")[0] + "}")
    table_entries[part_opt_s.split("_")[0] +"_"+ part_opt_s.split("_")[1]][d["name"]] = best

printed_string = 'ALGORITHM & '
for problem in PROBLEMS:
    printed_string += problem + " & "
print(printed_string + "\\\\")

for algorithm in sorted(table_entries['mushrooms_128'].keys()):
    printed_string = algorithm
    for problem in PROBLEMS: 
        for bs in ['128', '512']:
            printed_string += ' & ' + ('\\cmark' if table_entries[problem + '_' + bs][algorithm] == 'overlap' else '\\xmark')

    print(printed_string + ' \\\\')
             
