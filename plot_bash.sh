#!/bin/bash

datasets=("mushrooms" "ijcnn" "rcv1" "mnist" "fashion" "cifar10") 
batch_sizes=(128 512)
x_values=("time")
y_values=("train_loss") # "val_accuracy")
plot_numbers=(3)
logs_dir=("") # ENTER LOGS DIR

# Loop over each combination of parameters
for plot_number in "${plot_numbers[@]}"; do
    for dataset in "${datasets[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for y_value in "${y_values[@]}"; do
                if [ "$dataset" == "mushrooms" ] && [ "$batch_size" -eq 128 ]; then
                    python PlotResults/plotNew.py -p "$dataset" -x "time" -y "$y_value" -bs "$batch_size" -pn "$plot_number" -dl -path "$logs_dir" -log
                else
                    python PlotResults/plotNew.py -p "$dataset" -x "time" -y "$y_value" -bs "$batch_size" -pn "$plot_number" -dl -path "$logs_dir" -log
                fi
            done
        done
    done
done
