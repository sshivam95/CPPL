"""Validate all types."""
import json


def validate_json_file(json_param_file, solver):
    """
    Validate if the parameter file is of type json.

    :param json_param_file:
    :param solver:
    :return:
    """
    if json_param_file is None:
        json_file_name = "params_" + str(solver)

        with open(f"Configuration_Functions/{json_file_name}.json", "r") as file:
            data = file.read()
        params = json.loads(data)

        param_names = list(params.keys())

    else:
        params = json_param_file
        param_names = list(params.keys())
    return param_names, params
