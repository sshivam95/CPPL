"""Returning parameter set."""
import math


# pylint: disable=inconsistent-return-statements,too-many-branches
def log_space_convert(limit_number, param_set, json_param_file, exp=False):
    """Create a parameter set based on the maximum and minimum values in the json parameter file.

    Parameters
    ----------
    limit_number
    param_set
    json_param_file
    exp

    Returns
    -------
    param_set
    """
    if not exp:

        maxval_indices, minval_indices, param_names, params, to_delete = params_init(
            json_param_file
        )

        for index in sorted(to_delete, reverse=True):
            del param_names[index]

        update_max_min_values(
            limit_number, maxval_indices, minval_indices, param_names, params
        )

        for i, _ in enumerate(param_set):
            for j, _ in enumerate(maxval_indices):
                if float(param_set[i][maxval_indices[j]]) > 0:
                    param_set[i][maxval_indices[j]] = math.log(
                        float(param_set[i][maxval_indices[j]])
                    )
                elif float(param_set[i][maxval_indices[j]]) < 0:
                    param_set[i][maxval_indices[j]] = -math.log(
                        float(-param_set[i][maxval_indices[j]])
                    )
            for j, _ in enumerate(minval_indices):
                if float(param_set[i][minval_indices[j]]) < 0:
                    param_set[i][minval_indices[j]] = -math.log(
                        float(-param_set[i][minval_indices[j]])
                    )
                elif float(param_set[i][minval_indices[j]]) > 0:
                    param_set[i][minval_indices[j]] = math.log(
                        float(param_set[i][minval_indices[j]])
                    )

        return param_set

    if exp:

        maxval_indices, minval_indices, param_names, params, to_delete = params_init(
            json_param_file
        )

        update_max_min_values(
            limit_number, maxval_indices, minval_indices, param_names, params
        )

        for j, _ in enumerate(maxval_indices):
            if float(param_set[maxval_indices[j]]) > 0:
                param_set[maxval_indices[j]] = math.exp(
                    float(param_set[maxval_indices[j]])
                )
            elif float(param_set[maxval_indices[j]]) < 0:
                param_set[maxval_indices[j]] = -math.exp(
                    float(-param_set[maxval_indices[j]])
                )
        for j, _ in enumerate(minval_indices):
            if float(param_set[minval_indices[j]]) < 0:
                param_set[minval_indices[j]] = -math.exp(
                    float(-param_set[minval_indices[j]])
                )
            elif float(param_set[minval_indices[j]]) > 0:
                param_set[minval_indices[j]] = math.exp(
                    float(param_set[minval_indices[j]])
                )

        return param_set


# pylint: disable=bad-continuation
def update_max_min_values(
    limit_number, maxval_indices, minval_indices, param_names, params
):
    """Update the minimum and maximum value indices.

    Parameters
    ----------
    limit_number
    maxval_indices
    minval_indices
    param_names
    params
    """
    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "continuous":
            if params[param_names[i]]["maxval"] >= limit_number:
                maxval_indices.append(i)
            if params[param_names[i]]["minval"] <= -limit_number:
                minval_indices.append(i)
        if params[param_names[i]]["paramtype"] == "discrete":
            if params[param_names[i]]["maxval"] >= limit_number:
                maxval_indices.append(i)
            if params[param_names[i]]["minval"] <= -limit_number:
                minval_indices.append(i)
    for i, _ in enumerate(minval_indices):
        if minval_indices[i] in maxval_indices:
            del minval_indices[i]


def params_init(json_param_file):
    """Initialize parameters to be returned.

    Parameters
    ----------
    json_param_file

    Returns
    -------
    maxval_indices
    minval_indices
    param_names
    params
    to_delete
    """
    param_names = list(json_param_file.keys())
    params = json_param_file
    maxval_indices = []
    minval_indices = []
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
    return maxval_indices, minval_indices, param_names, params, to_delete
