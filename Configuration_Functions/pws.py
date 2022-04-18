import json
import jsonschema
from jsonschema import validate


def set_genes(solver, json_param_file):
    params = json_param_file
    param_names = list(json_param_file.keys())
    genes = [0 for i in range(len(param_names))]
    for i in range(len(param_names)):
        genes[i] = params[param_names[i]]["default"]

    return genes
