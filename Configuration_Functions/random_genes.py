"""Random Genes Utils."""
import json
import math
import random

import numpy as np
from numpy.random import choice


def get_all_min_and_max(solver_parameters):
    """
    Get the minimum and maximum values from the parameter file for the solver.

    :param solver_parameters: The parameter file in json format.
    :return: all minimum and maximum values from the file.
    """
    params = solver_parameters

    param_names = list(params.keys())

    all_min = []
    all_max = []

    for parameter_index, _ in enumerate(param_names):
        if params[param_names[parameter_index]]["paramtype"] == "categorical":
            if params[param_names[parameter_index]]["valtype"] == "str":
                values = params[param_names[parameter_index]]["values"]
                all_min.append(values[0])
                all_max.append(values[len(values) - 1])

            if params[param_names[parameter_index]]["valtype"] == "int":
                all_max.append(params[param_names[parameter_index]]["maxval"])
                all_min.append(params[param_names[parameter_index]]["minval"])

        elif params[param_names[parameter_index]]["paramtype"] == "continuous":
            all_max.append(params[param_names[parameter_index]]["maxval"])
            all_min.append(params[param_names[parameter_index]]["minval"])

        elif params[param_names[parameter_index]]["paramtype"] == "discrete":
            all_max.append(params[param_names[parameter_index]]["maxval"])
            all_min.append(params[param_names[parameter_index]]["minval"])

        elif params[param_names[parameter_index]]["paramtype"] == "binary":
            all_max.append(1)
            all_min.append(0)

    return all_min, all_max


# pylint: disable=too-many-nested-blocks,too-many-locals,too-many-branches,too-many-statements
# noinspection PyTypeChecker
def genes_set(solver, solver_parameters=None):
    """
    Return the gene set for a particular solver.

    :param solver: The solver instance for which the gene set is required.
    :param solver_parameters: The json parameter file.
    :return: genes set.
    """
    param_names, params = validate_json_file(solver_parameters, solver)

    genes = [0 for i, _ in enumerate(param_names)]

    for parameter_index, _ in enumerate(param_names):
        if params[param_names[parameter_index]]["paramtype"] == "categorical":

            # if params[param_names[parameter_index]]["valtype"] == "str":
            #     values = params[param_names[parameter_index]]["values"]
            #
            #     if "flag" in params[param_names[parameter_index]]:
            #         name_is_val = True
            #     else:
            #         name = param_names[parameter_index]
            #         name_is_val = False
            #     genes[parameter_index] = random.choice(values)
            #     if name_is_val:
            #         name = genes[parameter_index]

            if params[param_names[parameter_index]]["valtype"] == "int":
                max_val = params[param_names[parameter_index]]["maxval"]
                min_val = params[param_names[parameter_index]]["minval"]
                genes[parameter_index] = random.randint(min_val, max_val)

        elif params[param_names[parameter_index]]["paramtype"] == "continuous":
            default, max_val, min_val, splittable = split_by_default(
                index=parameter_index, param_names=param_names, params=params
            )

            if "distribution" in params[param_names[parameter_index]]:
                if params[param_names[parameter_index]]["distribution"] == "log":

                    (
                        high,
                        include_zero,
                        log_max_val,
                        log_on_neg,
                        log_on_pos,
                        low,
                        probab_pos,
                        probab_zero,
                        weights,
                    ) = get_log_distribution_params(
                        default=default,
                        parameter_index=parameter_index,
                        max_val=max_val,
                        min_val=min_val,
                        param_names=param_names,
                        params=params,
                        splittable=splittable,
                    )

                    if len(weights) == 3:
                        genes[parameter_index] = float(
                            choice([low, 0, high], 1, p=weights)
                        )
                    elif len(weights) == 2:
                        if not include_zero:
                            genes[parameter_index] = float(
                                choice([low, high], 1, p=weights)
                            )
                        else:
                            weights[1] = 1 - probab_zero
                            genes[parameter_index] = float(
                                choice([0, high], 1, p=weights)
                            )
                    if splittable:
                        if log_on_pos:
                            weights = [1 - probab_pos, probab_pos]
                        else:
                            weights = [0.5, 0.5]
                        genes[parameter_index] = float(
                            choice([low, high], 1, p=weights)
                        )
                    if not log_on_neg and not log_on_pos and min_val > 0:
                        genes[parameter_index] = float(
                            choice(
                                [
                                    math.exp(
                                        np.random.uniform(
                                            math.log(min_val), log_max_val, size=1
                                        )
                                    )
                                ],
                                1,
                            )[0]
                        )

            else:
                genes[parameter_index] = random.uniform(min_val, max_val)

        elif params[param_names[parameter_index]]["paramtype"] == "discrete":
            default, max_val, min_val, splittable = split_by_default(
                parameter_index, param_names, params
            )

            if "distribution" in params[param_names[parameter_index]]:
                if params[param_names[parameter_index]]["distribution"] == "log":

                    (
                        high,
                        include_zero,
                        log_max_val,
                        log_on_neg,
                        log_on_pos,
                        low,
                        probab_pos,
                        probab_zero,
                        weights,
                    ) = get_log_distribution_params(
                        default,
                        parameter_index,
                        max_val,
                        min_val,
                        param_names,
                        params,
                        splittable,
                    )

                    if len(weights) == 3:
                        genes[parameter_index] = int(
                            choice([low, 0, high], 1, p=weights)
                        )
                    elif len(weights) == 2:
                        if not include_zero:
                            genes[parameter_index] = int(
                                choice([low, high], 1, p=weights)
                            )
                        else:
                            weights[1] = 1 - probab_zero
                            genes[parameter_index] = int(
                                choice([0, high], 1, p=weights)
                            )
                    if splittable:
                        if log_on_pos:
                            weights = [1 - probab_pos, probab_pos]
                        else:
                            weights = [0.5, 0.5]
                        genes[parameter_index] = int(choice([low, high], 1, p=weights))

                    if not log_on_neg and not log_on_pos and min_val > 0:
                        genes[parameter_index] = int(
                            choice(
                                [
                                    math.exp(
                                        np.random.uniform(
                                            math.log(min_val), log_max_val, size=1
                                        )
                                    )
                                ],
                                1,
                            )
                        )

            else:
                genes[parameter_index] = random.randint(min_val, max_val)

        elif params[param_names[parameter_index]]["paramtype"] == "binary":
            # default = params[param_names[parameter_index]]["default"]
            genes[parameter_index] = random.randint(0, 1)

    return genes  # , params


# pylint: disable=bad-continuation,too-many-arguments,too-many-statements
def get_log_distribution_params(
        default, parameter_index, max_val, min_val, param_names, params, splittable
):
    """
    Return the parameters if the distribution is log.

    :param default:
    :param parameter_index:
    :param max_val:
    :param min_val:
    :param param_names:
    :param params:
    :param splittable:
    :return:
    """
    log_max_val = math.log(max_val)
    if min_val <= 0:
        log_min_val = 0
    else:
        log_min_val = math.log(min_val)
    log_on_pos = True
    log_on_neg = False
    probability_positive = None
    probability_zero = None
    include_zero = False
    weights = []
    if min_val <= 0:
        if "includezero" in params[param_names[parameter_index]]:
            if params[param_names[parameter_index]]["includezero"]:
                include_zero = True
                if "probabilityzero" in params[param_names[parameter_index]]:
                    if params[param_names[parameter_index]]["probabilityzero"]:
                        probability_zero = params[param_names[parameter_index]][
                            "probabilityzero"
                        ]
                    else:
                        # default probability if the probability
                        # for zero is not set in params_solver.json
                        probability_zero = 0.1
                    weights.append(probability_zero)

    if "log_on_pos" in params[param_names[parameter_index]]:
        if params[param_names[parameter_index]]["log_on_pos"]:
            log_on_pos = True
        if "probab_pos" in params[param_names[parameter_index]]:
            probability_positive = params[param_names[parameter_index]]["probab_pos"]
        else:
            if probability_zero and "probabneg" in params[param_names[parameter_index]]:
                probability_positive = (
                        1
                        - probability_zero
                        - params[param_names[parameter_index]]["probabneg"]
                )
            elif probability_zero:
                probability_positive = (1 - probability_zero) / 2
            else:
                probability_positive = 0.5
        weights.append(probability_positive)
    else:
        log_on_pos = False
    if min_val < 0:
        if "log_on_neg" in params[param_names[parameter_index]]:
            if params[param_names[parameter_index]]["log_on_neg"]:
                log_on_neg = True
                log_min_val = math.log(-min_val)
            if "probabneg" in params[param_names[parameter_index]]:
                probab_neg = params[param_names[parameter_index]]["probabneg"]
            else:
                if probability_zero:
                    probab_neg = 1 - probability_zero - probability_positive
                elif (
                        probability_zero
                        and probability_positive == (1 - probability_zero) / 2
                ):
                    probab_neg = (1 - probability_zero) / 2
                else:
                    probab_neg = 0.5
            weights = [probab_neg] + weights
        else:
            log_on_neg = False
    else:
        log_on_neg = False
    if log_on_pos:
        high = math.exp(np.random.uniform(0.000000001, log_max_val, size=1))
    else:
        high = random.uniform(0.000000001, max_val)
    if splittable:
        if default == 0:
            high = math.exp(np.random.uniform(default, log_max_val, size=1))
        else:
            high = math.exp(np.random.uniform(math.log(default), log_max_val, size=1))
    if log_on_neg:
        low = -math.exp(np.random.uniform(0.000000001, -log_min_val, size=1)) - min_val
    else:
        if splittable:
            low = random.uniform(min_val, default)
        else:
            low = random.uniform(min_val, 0.000000001)
    return (
        high,
        include_zero,
        log_max_val,
        log_on_neg,
        log_on_pos,
        low,
        probability_positive,
        probability_zero,
        weights,
    )


def split_by_default(index, param_names, params):
    """
    Check if the parameters names are splittable or not.

    :param index:
    :param param_names:
    :param params:
    :return:
    """
    max_val = params[param_names[index]]["maxval"]
    min_val = params[param_names[index]]["minval"]
    default = None
    if "splitbydefault" in params[param_names[index]]:
        if params[param_names[index]]["splitbydefault"]:
            default = params[param_names[index]]["default"]
            splittable = True
            return default, max_val, min_val, splittable
    return default, max_val, min_val, False


def one_hot_decode(
        genes, solver, param_value_dict=None, solver_parameters=None, reverse=False
):
    """
    Reverse One-Hot Encoding based on param_solver.json.

    :param genes:
    :param solver:
    :param param_value_dict:
    :param solver_parameters:
    :param reverse:
    :return:
    """
    param_names, params = validate_json_file(solver_parameters, solver)

    genes = list(genes)

    # one-hot encoding of previously one-hot decoded parameterization
    if (
            reverse
    ):  # One-Hot decoding here (one-hot parameters back to solver specific representation)

        pool = genes
        pool_set = []
        if len(genes[0]) == len(param_names):
            param_value_dict = {}

            for j, _ in enumerate(pool):
                next_params = list(pool[j])
                params = solver_parameters
                originals_ind_to_delete = []
                one_hot_addition = []
                for i, _ in enumerate(param_names):
                    param_value_dict[param_names[i]] = next_params[i]
                    if params[param_names[i]]["paramtype"] == "categorical":
                        if params[param_names[i]]["valtype"] == "int":
                            min_val = params[param_names[i]]["minval"]
                            max_val = params[param_names[i]]["maxval"]
                            value_range = int(max_val) - int(min_val) + 1
                            values = [min_val + j for j in range(value_range)]
                            index = int(next_params[i]) - min_val

                            if len(values) > 2:
                                # The categorical Parameter needs One-Hot Encoding
                                # -> append One-Hot Vectors and delete original elements
                                one_hot = [0 for j, _ in enumerate(values)]
                                one_hot[index] = 1
                                originals_ind_to_delete.append(i)
                                one_hot_addition += one_hot

                                param_value_dict[param_names[i]] = None

                new_params = []

                for key in param_value_dict:
                    if param_value_dict[key] is not None:
                        new_params.append(param_value_dict[key])

                new_params += one_hot_addition

                pool_set.append(new_params)
        else:
            pool_set = pool

        return pool_set

    if len(genes) != len(param_names):
        one_hot_tail = []
        one_hot_vector_ranges = []
        real_vector = []

        for i, _ in enumerate(param_names):
            if params[param_names[i]]["paramtype"] == "categorical":
                if params[param_names[i]]["valtype"] == "int":
                    min_val = params[param_names[i]]["minval"]
                    max_val = params[param_names[i]]["maxval"]
                    if (max_val - min_val + 1) > 2:
                        one_hot_vector_ranges.append(max_val - min_val + 1)
                        real_vector.append(
                            [min_val + t for t in range(max_val - min_val + 1)]
                        )
                        one_hot_part = [0 for _ in range(max_val - min_val + 1)]
                        one_hot_tail.append(one_hot_part)

        one_hot_tail_len = 0
        for i, _ in enumerate(one_hot_tail):
            one_hot_tail_len += len(one_hot_tail[i])

        one_hot_values = genes[len(genes) - one_hot_tail_len:]

        cat_int_values = []

        for i, _ in enumerate(one_hot_tail):
            cat_int_index = one_hot_values[: len(one_hot_tail[i])].index(1)
            cat_int_values.append(real_vector[i][cat_int_index])
            del one_hot_values[: len(one_hot_tail[i])]

        values_k = []
        for i, _ in enumerate(cat_int_values):
            values_k.append(cat_int_values[i])

        new_genes = []
        insert_count = 0

        k = 0  # Length of the subset Line 9 CPPL
        for key in param_value_dict:
            if param_value_dict[key] is None:
                int_value = cat_int_values.pop(0)
                new_genes.append(int_value)
                insert_count += 1
                k += 1
            else:
                new_genes.append(genes[k - insert_count])
                k += 1

        return new_genes
    return genes


def get_params_string_from_numeric_params(genes, solver, solver_parameters=None):
    """
    Transform string categorical back to string based on params_solver.json.

    :param genes:
    :param solver:
    :param solver_parameters:
    :return:
    """
    param_names, params = validate_json_file(solver_parameters, solver)

    genes = list(genes)

    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":
            if params[param_names[i]]["valtype"] == "str":
                values = params[param_names[i]]["values"]
                string_value = values[int(genes[i])]
                genes[i] = str(string_value)

    return genes


def validate_json_file(solver_parameters, solver):
    """
    Validate if the parameter file is of type json.

    :param solver_parameters:
    :param solver:
    :return:
    """
    if solver_parameters is None:
        json_file_name = "params_" + str(solver)

        with open(f"Configuration_Functions/{json_file_name}.json", "r") as file:
            data = file.read()
        params = json.loads(data)

        param_names = list(params.keys())

    else:
        params = solver_parameters
        param_names = list(params.keys())
    return param_names, params
