import json
import os
from datetime import datetime
import csv
import copy
import itertools

def init_logs_folder(savedir_base):
    exp_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    savedir = os.path.join(savedir_base, exp_date)

    os.makedirs(savedir, exist_ok=True)

    return savedir


def init_exp_log_folder(savedir, exp_dict, exp_fields_logs):

    exp_str = ""
    for exp_field in exp_fields_logs:
        if "opt." in exp_field:
            opt_field = exp_field.split(".")[1]
            if opt_field in exp_dict['opt'].keys():
                exp_str += str(exp_dict['opt'][opt_field]) + "_"
        else:
            exp_str += str(exp_dict[exp_field]) + "_"

    exp_str = exp_str[:-1]
    
    expdir = os.path.join(savedir, exp_str)
    os.makedirs(expdir, exist_ok=True)

    return expdir

def init_csv(savedir, csv_name, col_names):
    csv_path = os.path.join(savedir, csv_name)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(col_names)

def append_row_csv(savedir, csv_name, row):
    csv_path = os.path.join(savedir, csv_name)

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(row)

def append_rows_csv(savedir, csv_name, rows):
    csv_path = os.path.join(savedir, csv_name)

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerows(rows)

def save_json(fname, data, makedirs=True):
    """
    # From Haven utils
    """

    # turn fname to string in case it is a Path object
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def cartesian_exp_group(exp_config, remove_none=False):
    """
    # From Haven utils
    """

    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), values)) for values in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        # remove hparams with None
        if remove_none:
            to_remove = []
            for k, v in exp_dict.items():
                if v is None:
                    to_remove += [k]
            for k in to_remove:
                del exp_dict[k]
        exp_list += [exp_dict]

    return exp_list


