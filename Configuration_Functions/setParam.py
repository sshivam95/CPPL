import json
import jsonschema
from jsonschema import validate


def set_params(contender, genes, solver, json_param_file, return_it=False):

    paramNames = list(json_param_file.keys())
    params = json_param_file

    parameter_set = [0 for i in range(len(genes))]

    for i in range(len(genes)):
        if "flag" in params[paramNames[i]]:
            parameter_set[i] = str(genes[i])
        else:
            if json_param_file[paramNames[i]]["paramtype"] == "discrete":
                parameter_set[i] = str(paramNames[i]) + str(int(genes[i]))
            else:
                parameter_set[i] = str(paramNames[i]) + str(genes[i])

    with open("ParamPool/" + str(contender), "w") as file:
        print(" ".join(parameter_set), file=file)

    if return_it:
        return parameter_set
