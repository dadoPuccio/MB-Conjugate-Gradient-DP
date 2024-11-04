import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plotUtils import *

parser = argparse.ArgumentParser()

parser.add_argument('-dir', '--exp_dir')
parser.add_argument('-p', '--dataset', default='ijcnn')
parser.add_argument('-alg', '--algorithm', default='sgdm')
parser.add_argument('-bs', '--batch_size', default=128, type=int)
parser.add_argument('-beta', '--beta', default=0.9, type=float)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-angle', action='store_true', help="Plot avg. Angles")
group.add_argument('-count', action='store_true', help="Plot count non-descent")

args = parser.parse_args()

title_params = ["dataset", "batch_size"]

x_lim = [None, None]
y_lim = [-1, 1]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
markers = ['.', ',', 'v', '.', '^'] 

counter = 0
overall_counter = []

plt.figure(figsize=(7, 5))
plt.rcParams.update({'font.size': 16})

for entry in sorted(os.listdir(args.exp_dir)):

    params = read_json(os.path.join(args.exp_dir, entry, "params.json"))

    if params["batch_size"] != args.batch_size or \
       params['dataset'] != args.dataset or \
       params["opt"]['beta'] != args.beta :
        continue 

    print("Plotting ", entry)

    reference_params = params
    
    epoch_stats = read_csv(os.path.join(args.exp_dir, entry, "epoch_stats.csv"))

    updates_per_epoch = epoch_stats['n_forwards'][0]

    for overlap_percentage in params["opt"]['overlap_percentages']:

        x = epoch_stats['epoch']

        if args.angle:
            field = 'avg_' + args.algorithm + '_angle_' + str(overlap_percentage)
            y = [v for v in epoch_stats[field]]

        elif args.count:
            field = args.algorithm + '_count_' + str(overlap_percentage)
            overall_counter.append(sum(epoch_stats[field]))
            y = [v / updates_per_epoch for v in epoch_stats[field]]
        
        label = str(overlap_percentage) + "%" 

        plt.plot(x, y, label=label, marker=markers[counter], color=colors[counter])

        counter += 1

if args.count:
    counter_latex_str = ''
    for c in overall_counter:
        counter_latex_str += str(int(c)) + " & "

    print(counter_latex_str)

handles, labels = plt.gca().get_legend_handles_labels()
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

plt.legend(handles, labels, loc='lower right') #, loc='center left', bbox_to_anchor=(1, 0.5))


plt.xlabel("epochs")
if args.angle:
    plt.ylabel("Avg. Angle [deg]")
elif args.count:
    plt.ylabel("Ratio of non-descent momentum directions")

if args.algorithm == 'sgdm':
    title = "" # Overlap Analysis    SGD+M    "
elif args.algorithm == 'mag':
    title = "" # Overlap Analysis    MAG    "

for k in title_params:
    if "." in k:
        k1, k2 = k.split(".")
        title += k2 + ": " + str(reference_params[k1][k2]) + "    "
    else:
        title += k + ": " + str(reference_params[k]) + "    "

plt.title(title) 

plt.tight_layout()  # Adjust layout to make room
# plt.show()  # Display the plot

pp = PdfPages("{}{}_{}_{}.pdf".format("Plots/", args.algorithm, args.batch_size, "angle" if args.angle else "count"))
plt.savefig(pp, format='pdf')
pp.close()


