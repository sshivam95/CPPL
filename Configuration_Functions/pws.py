import json
import jsonschema
from jsonschema import validate

def set_genes(solver,json_param_file):

    params = json_param_file

    paramNames = list(json_param_file.keys())

    genes = [0 for i in range(len(paramNames))]


    for i in range(len(paramNames)):
        genes[i] = params[paramNames[i]]['default']


    return genes