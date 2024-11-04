import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plotUtils import *
from plotConfigs import *

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--problem')
parser.add_argument('-pn', '--plot_number', default=0, type=int)
parser.add_argument('-bs', '--batch_size', default=128, type=int)
parser.add_argument('-x', '--x_axis', default="epoch")
parser.add_argument('-y', '--y_axis', default="train_loss")
parser.add_argument('-path', '--path', default=None)

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-log', action='store_true', help="Log y axis")
group.add_argument('-lin', action='store_true', help="Linear y axis")

parser.add_argument('-dl', '--display_legend', action='store_true', default=False)
parser.add_argument('-leg_o', '--legend_overlap', action='store_true', default=False)
parser.add_argument('-leg_c', '--legend_c_line_search', action='store_true', default=False)
parser.add_argument('-leg_all', '--legend_all', action='store_true', default=False)

args = parser.parse_args()

custom_filter_convex, custom_filter_non_convex, colors, markers = configs_wrapper(args.plot_number, args.problem, args.batch_size)

if args.path is not None:
    
    base_dir = args.path
    cg_dir = base_dir
    ref_dir = base_dir
    if args.problem in ['ijcnn', 'rcv1', 'mushrooms']:
        custom_filter = custom_filter_convex
    else:
        custom_filter = custom_filter_non_convex

else:
    raise ValueError("Please specify path")

x_lim = [None, None]
y_lim = [-1, 1]

to_be_plotted = {}

plt.figure(figsize=(7, 5))
plt.rcParams.update({'font.size': 16})

for entry in [os.path.join(cg_dir, e) for e in sorted(os.listdir(cg_dir))] + [os.path.join(ref_dir, e) for e in sorted(os.listdir(ref_dir))]:

    params = read_json(os.path.join(entry, "params.json"))

    if params["dataset"] != args.problem or params["batch_size"] != args.batch_size:
        continue

    if filter_removed(params, custom_filter):
        # print("Removed", params["opt"])
        continue

    # if params['opt']['name'] == 'sgd_armijo' and params['runs'] == 0:
    #     continue

    print("Plotting", params["opt"]["name"], os.path.join(entry))

    reference_params = params
    
    epoch_stats = read_csv(os.path.join(entry, "epoch_stats.csv"))
    minibatch_stats = read_csv(os.path.join(entry, "minibatch_stats.csv"))

    if args.x_axis in epoch_stats and args.y_axis in epoch_stats:
        x = epoch_stats[args.x_axis]
        y = epoch_stats[args.y_axis]

    elif args.x_axis in minibatch_stats and args.y_axis in minibatch_stats:
        x = minibatch_stats[args.x_axis]
        y = minibatch_stats[args.y_axis]

    else:
        raise FileNotFoundError("Unable to read the x_axis / y_axis specified, perhaps wrong axis name in input?")
    
    current_key = str({k: v for k, v in params['opt'].items() if k not in ['max_beta', 'max_eta'] }) + "_" + str(params["overlap_batches"])

    if current_key not in to_be_plotted:
        to_be_plotted[current_key] = {}

    to_be_plotted[current_key][params["runs"]] = {'x': x, 'y': y, 'params': params}

counter = 0
max_x = -1

# plt.axhline(y=1e-5, color='g', linestyle='--', linewidth=1)

for opt_signature, runs in sorted(to_be_plotted.items()):

    print(opt_signature)
    
    max_len = max([len(r['x']) for r in runs.values()])

    all_x = []
    all_y = []
    for r in runs.values():
        all_x.append(np.pad(r['x'], (0, max_len - len(r['x'])), 'linear_ramp', end_values=(r['x'][-1] + (max_len - len(r['x'])) * np.mean([r['x'][i+1] - r['x'][i] for i in range(len(r['x']) - 1)]),)))
        all_y.append(np.pad(r['y'], (0, max_len - len(r['y'])), 'constant'))

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    mean_x = np.mean(all_x, axis=0)

    if max(mean_x) > max_x:
        max_x = max(mean_x)

    mean_y = np.mean(all_y, axis=0)
    std_y = np.std(all_y, axis=0)

    min_y = np.amin(all_y, axis=0)
    max_y = np.amax(all_y, axis=0)

    label = r['params']['opt']['name']

    if label == 'sgd':
        label = 'SGD+M'
    elif label == 'adam':
        label = 'Adam'
    elif label == "conjugate_gradient": 
        if args.plot_number == 0:
            label = r['params']['opt']['cg_mode']
        elif args.plot_number == 1:
            if r['params']['opt']['eta_mode'] == 'vaswani':
                label = 'heuristic'     
            else:
                label = r['params']['opt']['eta_mode']
        elif args.plot_number == 2:
            label_map = {"grad": "gradient", "inv": "inversion", "qps": "subspace", "clip": "clipping"}
            label = label_map[r['params']['opt']['dir_recovery_mode']]
        else:
            label = 'MBCG_' + r['params']['opt']['cg_mode'] #  + r['params']['opt']['dir_recovery_mode']

    if args.legend_overlap:
        label += " (" + str(r['params']['overlap_batches']) + ")"
    
    if args.legend_all:
        label += str(r['params']['opt']) 
    
    if args.legend_c_line_search:
        label += str(r['params']['opt']['c'])
        label += str(r['params']['opt']['eta_mode'])
    
    if args.log:
        plt.semilogy(mean_x, mean_y, label=label, marker=markers[counter], color=colors[counter])
    elif args.lin:
        plt.plot(mean_x, mean_y, label=label, marker=markers[counter], color=colors[counter])
    else:
        print("error")
    plt.fill_between(mean_x, min_y, 
                     max_y, 
                     color=colors[counter], 
                     alpha=0.2)


    counter += 1


handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

if args.display_legend:
    plt.legend(handles, labels, loc='upper right')# , loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(True)
plt.xlim(0, max_x)
# plt.ylim(0.6, 0.9)

plt.xlabel(args.x_axis + " [s]" if args.x_axis == 'time' else args.x_axis)
plt.ylabel(args.y_axis)

title_params = ["dataset", "model", "batch_size"]
title = ''
for k in title_params:
    if "." in k:
        k1, k2 = k.split(".")
        title += k2 + ": " + str(reference_params[k1][k2]) + "    "
    else:
        title += str(reference_params[k]) + " | " # k + ": " + 

title = title[:-2]
plt.title(title) 

plt.tight_layout() 
# plt.show()  

if args.plot_number == 0:
    additional_label = 'beta'
elif args.plot_number == 1:
    additional_label = 'alpha0'
elif args.plot_number == 2:
    additional_label = 'dk'
else:
    additional_label = ''

pp = PdfPages("{}{}_{}_{}_{}_{}.pdf".format("Plots/", args.problem, args.x_axis, args.y_axis, args.batch_size, additional_label))
plt.savefig(pp, format='pdf')
pp.close()
