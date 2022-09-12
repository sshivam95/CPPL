import math

import numpy as np


def log_space_convert(
    limit_number: int, param_set: np.ndarray, solver_parameter: dict, exp: bool = False
) -> np.ndarray:
    """
    Convert the parameter set based on the maximum and minimum values in the json parameter file to logarithm space.

    :param limit_number: Upper and lower limit of the minimum and maximum values from the solver parameter set.
    :param param_set: Parameters to be converted.
    :param solver_parameter: Parameters of the solver.
    :param exp: ??
    :return:  A log converted parameter set.
    """

    max_val_indices, min_val_indices, param_names, params, to_delete = _params_init(
        solver_parameters=solver_parameter
    )

    if not exp:

        for index in sorted(to_delete, reverse=True):
            del param_names[index]

        update_max_min_values(
            limit_number=limit_number,
            max_val_indices=max_val_indices,
            min_val_indices=min_val_indices,
            param_names=param_names,
            params=params,
        )

        for i, _ in enumerate(param_set):
            for j, _ in enumerate(max_val_indices):
                if float(param_set[i][max_val_indices[j]]) > 0:
                    param_set[i][max_val_indices[j]] = math.log(
                        float(param_set[i][max_val_indices[j]])
                    )
                elif float(param_set[i][max_val_indices[j]]) < 0:
                    param_set[i][max_val_indices[j]] = -math.log(
                        float(-param_set[i][max_val_indices[j]])
                    )
            for j, _ in enumerate(min_val_indices):
                if float(param_set[i][min_val_indices[j]]) < 0:
                    param_set[i][min_val_indices[j]] = -math.log(
                        float(-param_set[i][min_val_indices[j]])
                    )
                elif float(param_set[i][min_val_indices[j]]) > 0:
                    param_set[i][min_val_indices[j]] = math.log(
                        float(param_set[i][min_val_indices[j]])
                    )

        return param_set

    if exp:  # Question: What is exp? Use?

        update_max_min_values(
            limit_number=limit_number,
            max_val_indices=max_val_indices,
            min_val_indices=min_val_indices,
            param_names=param_names,
            params=params,
        )

        for j, _ in enumerate(max_val_indices):
            if float(param_set[max_val_indices[j]]) > 0:
                param_set[max_val_indices[j]] = math.exp(
                    float(param_set[max_val_indices[j]])
                )
            elif float(param_set[max_val_indices[j]]) < 0:
                param_set[max_val_indices[j]] = -math.exp(
                    float(-param_set[max_val_indices[j]])
                )
        for j, _ in enumerate(min_val_indices):
            if float(param_set[min_val_indices[j]]) < 0:
                param_set[min_val_indices[j]] = -math.exp(
                    float(-param_set[min_val_indices[j]])
                )
            elif float(param_set[min_val_indices[j]]) > 0:
                param_set[min_val_indices[j]] = math.exp(
                    float(param_set[min_val_indices[j]])
                )

        return param_set


def update_max_min_values(
    limit_number: int,
    max_val_indices: list,
    min_val_indices: list,
    param_names: list,
    params: dict,
) -> None:
    """
    Update the minimum and maximum values based on the parameters of the solver.

    :param limit_number: Upper and lower limit of the minimum and maximum values from the solver parameter set.
    :param max_val_indices: List of indices of the maximum values in the parameter set of the solver.
    :param min_val_indices: List of indices of the minimum values in the parameter set of the solver.
    :param param_names: List of keys in the parameter set of the solver.
    :param params: Parameter set of the solver.
    """
    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "continuous":
            if params[param_names[i]]["maxval"] >= limit_number:
                max_val_indices.append(i)
            if params[param_names[i]]["minval"] <= -limit_number:
                min_val_indices.append(i)
        if params[param_names[i]]["paramtype"] == "discrete":
            if params[param_names[i]]["maxval"] >= limit_number:
                max_val_indices.append(i)
            if params[param_names[i]]["minval"] <= -limit_number:
                min_val_indices.append(i)
    for i, _ in enumerate(min_val_indices):
        if min_val_indices[i] in max_val_indices:
            del min_val_indices[i]


def _params_init(solver_parameters: dict) -> (list, list, list, dict, list):
    """
    Initialize parameters attributes.

    :param solver_parameters: Solver's parameters

    :return max_val_indices: List of indices of the maximum values in the parameter set of the solver.
    :return min_val_indices: List of indices of the minimum values in the parameter set of the solver.
    :return param_names: List of keys in the parameter set of the solver.
    :return params: Parameter set of the solver.
    :return to_delete: Parameters to be deleted.
    """
    param_names = list(solver_parameters.keys())
    params = solver_parameters
    max_val_indices = []
    min_val_indices = []
    to_delete = []
    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":
            if params[param_names[i]]["valtype"] == "int":
                min_val = params[param_names[i]]["minval"]
                max_val = params[param_names[i]]["maxval"]
                value_range = max_val - min_val + 1
                values = [min_val + j for j in range(value_range)]
                if len(values) > 2:
                    to_delete.append(i)
    return max_val_indices, min_val_indices, param_names, params, to_delete
