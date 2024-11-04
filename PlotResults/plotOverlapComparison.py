import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plotUtils import *

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--problem')
parser.add_argument('-dir', '--exp_dir')
parser.add_argument('-alg', '--algorithm', default='sgdm') # or "adam"
parser.add_argument('-x', '--x_axis', default="time")
parser.add_argument('-y', '--y_axis', default="train_loss")
parser.add_argument('-bs', '--batch_size', default=128, type=int)

args = parser.parse_args()

custom_filter = {"sgd": {"lr": 0.001, "momentum": 0.9}, 
                 "adam": {"lr": 0.00001}}


colors = ['#1f77b4', '#ff7f0e']
markers = ['.', ','] 

exp_dir = args.exp_dir

plt.figure(figsize=(7, 5))
plt.rcParams.update({'font.size': 16})

to_be_plotted = {}

for entry in sorted(os.listdir(exp_dir)):

    params = read_json(os.path.join(exp_dir, entry, "params.json"))

    print(params)

    if params["dataset"] != args.problem or params["batch_size"] != args.batch_size:
        continue

    if params['opt']['name'] not in ['sgd', 'adam', 'sls_zhangNM_polyak', 'msl_sgdm_c']:
        continue

    if params['opt']['name'] == 'sgd' and args.algorithm != 'sgdm':
        continue
    elif params['opt']['name'] == 'adam' and args.algorithm != 'adam':
        continue
    elif params['opt']['name'] == 'sls_zhangNM_polyak' and args.algorithm != 'ponos':
        continue
    elif params['opt']['name'] == 'msl_sgdm_c' and args.algorithm != 'msl_sgdm':
        continue


    if filter_removed(params, custom_filter):
        # print("Removed", params["opt"])
        continue
    
    print("Selected", params['opt'])

    epoch_stats = read_csv(os.path.join(exp_dir, entry, "epoch_stats.csv"))
    minibatch_stats = read_csv(os.path.join(exp_dir, entry, "minibatch_stats.csv"))

    if args.x_axis in epoch_stats and args.y_axis in epoch_stats:
        x = epoch_stats[args.x_axis]
        y = epoch_stats[args.y_axis]

    elif args.x_axis in minibatch_stats and args.y_axis in minibatch_stats:
        x = minibatch_stats[args.x_axis]
        y = minibatch_stats[args.y_axis]

    else:
        raise FileNotFoundError("Unable to read the x_axis / y_axis specified, perhaps wrong axis name in input?")
    
    current_key = params["opt"]["name"] + "_" + str(params["overlap_batches"])

    if current_key not in to_be_plotted:
        to_be_plotted[current_key] = {}

    to_be_plotted[current_key][params["runs"]] = {'x': x, 'y': y, 'params': params}


counter = 0

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

    mean_y = np.mean(all_y, axis=0)
    std_y = np.std(all_y, axis=0)

    min_y = np.amin(all_y, axis=0)
    max_y = np.amax(all_y, axis=0)

    plt.semilogy(mean_x, mean_y, label="overlap" if r['params']['overlap_batches'] else "no-overlap", marker=markers[counter], color=colors[counter])
    # plt.plot(x, y, label=plt_legend, marker=markers[counter], color=colors[counter])
    plt.fill_between(mean_x, min_y, 
                     max_y, 
                     color=colors[counter], 
                     alpha=0.2)


    counter += 1


handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

plt.legend(handles, labels) #, loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid(True)
plt.xlim(0, None)

plt.xlabel(args.x_axis  + ' [s]' if args.x_axis == 'time' else args.x_axis)
plt.ylabel(args.y_axis)

title = '' # 'Overlap Analysis     '
if args.algorithm == 'sgdm':
    title += "SGD+M    "
    title_params = ["opt.lr", "opt.momentum"]
elif args.algorithm == 'adam':
    title += "Adam    "
    title_params = ["opt.lr"]
elif args.algorithm == 'ponos':
    title += "PoNoS    "
    title_params = ["opt.c_step"]
elif args.algorithm == 'msl_sgdm':
    title += "MSL_SGDM    "
    title_params = ["opt.beta"]

title_params += ["dataset", "batch_size"]

for k in title_params:
    if "." in k:
        k1, k2 = k.split(".")
        if k2 == 'momentum':
            k2_title = 'beta'
        elif k2 == 'c_step':
            k2_title = 'c_p'
        else:
            k2_title = k2
        title += k2_title + ": " + str(r['params'][k1][k2]) + "    "
    else:
        title += str(r['params'][k]) + "    "   # k + ": " + 


plt.title(title) 

plt.tight_layout()  # Adjust layout to make room
# plt.show()  # Display the plot

pp = PdfPages("{}{}_{}_{}_{}.pdf".format("Plots/", args.problem, args.algorithm, args.batch_size, "overlapComp"))
plt.savefig(pp, format='pdf')
pp.close()
