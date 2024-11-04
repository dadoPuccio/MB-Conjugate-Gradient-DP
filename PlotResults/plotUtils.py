import json
import csv

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def read_csv(path):
    with open(path, mode='r', newline='') as file:
        reader = csv.reader(file)

        data = {}

        for title in next(reader):
            data[title] = []

        for row in reader:
            for v, title in zip(row, data.keys()):
                if "tensor" in v:
                    data[title].append(float(v.replace("(", " ").replace(")","").split(" ")[1]))
                elif "False" in v or "True" in v:
                    data[title].append(bool(v))
                else:
                    data[title].append(float(v))

        return data
    
def filter_removed(params, custom_filter):
    if params["opt"]["name"] not in custom_filter.keys():
        return True # remove in case the opt is not in the filter

    if custom_filter[params["opt"]["name"]] == 'all':
        return False

    for k, v in custom_filter[params["opt"]["name"]].items():
        if k == 'overlap_batches': 
            if params['overlap_batches'] != v:
                return True

        elif type(v) == list:
            if params["opt"][k] not in v:
                return True
            
        else:
            # print(k, v, params["opt"][k])
            if params["opt"][k] != v:
                return True
        
    return False
