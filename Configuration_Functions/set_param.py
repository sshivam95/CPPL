import json
import jsonschema
from jsonschema import validate


def set_contender_params(contender_index, genes, solver_parameters, return_it=False):

    param_names = list(solver_parameters.keys())
    params = solver_parameters

    parameter_set = [0 for i in range(len(genes))]

    for i in range(len(genes)):
        if "flag" in params[param_names[i]]:
            parameter_set[i] = str(genes[i])
        else:
            if solver_parameters[param_names[i]]["paramtype"] == "discrete":
                parameter_set[i] = str(param_names[i]) + str(int(genes[i]))
            else:
                parameter_set[i] = str(param_names[i]) + str(genes[i])

    with open("ParamPool/" + str(contender_index), "w") as file:
        print(" ".join(parameter_set), file=file)

    if return_it:
        return parameter_set
