"""Random Genes Utils."""
import math
import random

import numpy as np
from numpy.random import choice
from validation import validate_json_file


def get_all_min_and_max(json_param_file):
    """
    Get the minimum and maximum values from the parameter file for the solver.

    :param json_param_file: The parameter file in json format.
    :return: all minimum and maximum values from the file.
    """
    params = json_param_file

    param_names = list(params.keys())

    all_min = []
    all_max = []

    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":
            if params[param_names[i]]["valtype"] == "str":
                values = params[param_names[i]]["values"]
                all_min.append(values[0])
                all_max.append(values[len(values) - 1])

            if params[param_names[i]]["valtype"] == "int":
                all_max.append(params[param_names[i]]["maxval"])
                all_min.append(params[param_names[i]]["minval"])

        elif params[param_names[i]]["paramtype"] == "continuous":
            all_max.append(params[param_names[i]]["maxval"])
            all_min.append(params[param_names[i]]["minval"])

        elif params[param_names[i]]["paramtype"] == "discrete":
            all_max.append(params[param_names[i]]["maxval"])
            all_min.append(params[param_names[i]]["minval"])

        elif params[param_names[i]]["paramtype"] == "binary":
            all_max.append(1)
            all_min.append(0)

    return all_min, all_max


# pylint: disable=too-many-nested-blocks,too-many-locals,too-many-branches,too-many-statements
def genes_set(solver, json_param_file=None):
    """
    Return the gene set for a particular solver.

    :param solver: The solver instance for which the gene set is required.
    :param json_param_file: The json parameter file.
    :return: genes set.
    """
    param_names, params = validate_json_file(json_param_file, solver)

    genes = [0 for i, _ in enumerate(param_names)]

    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":

            # if params[param_names[i]]["valtype"] == "str":
            #     values = params[param_names[i]]["values"]
            #
            #     if "flag" in params[param_names[i]]:
            #         name_is_val = True
            #     else:
            #         name = param_names[i]
            #         name_is_val = False
            #     genes[i] = random.choice(values)
            #     if name_is_val:
            #         name = genes[i]

            if params[param_names[i]]["valtype"] == "int":
                max_val = params[param_names[i]]["maxval"]
                min_val = params[param_names[i]]["minval"]
                genes[i] = random.randint(min_val, max_val)

        elif params[param_names[i]]["paramtype"] == "continuous":
            default, max_val, min_val, splittable = split_by_default(
                i, param_names, params
            )

            if "distribution" in params[param_names[i]]:
                if params[param_names[i]]["distribution"] == "log":

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
                        default, i, max_val, min_val, param_names, params, splittable
                    )

                    if len(weights) == 3:
                        genes[i] = float(choice([low, 0, high], 1, p=weights))
                    elif len(weights) == 2:
                        if not include_zero:
                            genes[i] = float(choice([low, high], 1, p=weights))
                        else:
                            weights[1] = 1 - probab_zero
                            genes[i] = float(choice([0, high], 1, p=weights))
                    if splittable:
                        if log_on_pos:
                            weights = [1 - probab_pos, probab_pos]
                        else:
                            weights = [0.5, 0.5]
                        genes[i] = float(choice([low, high], 1, p=weights))
                    if not log_on_neg and not log_on_pos and min_val > 0:
                        genes[i] = float(
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
                genes[i] = random.uniform(min_val, max_val)

        elif params[param_names[i]]["paramtype"] == "discrete":
            default, max_val, min_val, splittable = split_by_default(
                i, param_names, params
            )

            if "distribution" in params[param_names[i]]:
                if params[param_names[i]]["distribution"] == "log":

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
                        default, i, max_val, min_val, param_names, params, splittable
                    )

                    if len(weights) == 3:
                        genes[i] = int(choice([low, 0, high], 1, p=weights))
                    elif len(weights) == 2:
                        if not include_zero:
                            genes[i] = int(choice([low, high], 1, p=weights))
                        else:
                            weights[1] = 1 - probab_zero
                            genes[i] = int(choice([0, high], 1, p=weights))
                    if splittable:
                        if log_on_pos:
                            weights = [1 - probab_pos, probab_pos]
                        else:
                            weights = [0.5, 0.5]
                        genes[i] = int(choice([low, high], 1, p=weights))

                    if not log_on_neg and not log_on_pos and min_val > 0:
                        genes[i] = int(
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
                genes[i] = random.randint(min_val, max_val)

        elif params[param_names[i]]["paramtype"] == "binary":
            default = params[param_names[i]]["default"]
            genes[i] = random.randint(0, 1)

    return genes  # , params


# pylint: disable=bad-continuation,too-many-arguments,too-many-statements
def get_log_distribution_params(
    default, i, max_val, min_val, param_names, params, splittable
):
    """
    Return the parameters if the distribution is log.

    :param default:
    :param i:
    :param max_val:
    :param min_val:
    :param param_names:
    :param params:
    :param splittable:
    :return:
    """
    log_max_val = math.log(max_val)
    log_min_val = math.log(min_val)
    log_on_pos = True
    log_on_neg = False
    probab_pos = None
    probab_zero = None
    include_zero = False
    weights = []
    if min_val <= 0:
        if "includezero" in params[param_names[i]]:
            if params[param_names[i]]["includezero"]:
                include_zero = True
                if "probabilityzero" in params[param_names[i]]:
                    if params[param_names[i]]["probabilityzero"]:
                        probab_zero = params[param_names[i]]["probabilityzero"]
                    else:
                        # default probability if the probability
                        # for zero is not set in params_solver.json
                        probab_zero = 0.1
                    weights.append(probab_zero)

    if "log_on_pos" in params[param_names[i]]:
        if params[param_names[i]]["log_on_pos"]:
            log_on_pos = True
        if "probab_pos" in params[param_names[i]]:
            probab_pos = params[param_names[i]]["probab_pos"]
        else:
            if probab_zero and "probabneg" in params[param_names[i]]:
                probab_pos = 1 - probab_zero - params[param_names[i]]["probabneg"]
            elif probab_zero:
                probab_pos = (1 - probab_zero) / 2
            else:
                probab_pos = 0.5
        weights.append(probab_pos)
    else:
        log_on_pos = False
    if min_val < 0:
        if "log_on_neg" in params[param_names[i]]:
            if params[param_names[i]]["log_on_neg"]:
                log_on_neg = True
                log_min_val = math.log(-min_val)
            if "probabneg" in params[param_names[i]]:
                probab_neg = params[param_names[i]]["probabneg"]
            else:
                if probab_zero:
                    probab_neg = 1 - probab_zero - probab_pos
                elif probab_zero and probab_pos == (1 - probab_zero) / 2:
                    probab_neg = (1 - probab_zero) / 2
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
        probab_pos,
        probab_zero,
        weights,
    )


def split_by_default(i, param_names, params):
    """
    Check if the parameters names are splittable or not.

    :param i:
    :param param_names:
    :param params:
    :return:
    """
    max_val = params[param_names[i]]["maxval"]
    min_val = params[param_names[i]]["minval"]
    default = None
    if "splitbydefault" in params[param_names[i]]:
        if params[param_names[i]]["splitbydefault"]:
            default = params[param_names[i]]["default"]
            splittable = True
            return default, max_val, min_val, splittable
    return default, max_val, min_val, False


def one_hot_decode(
    genes, solver, param_value_dict=None, json_param_file=None, reverse=False
):
    """
    Reverse One-Hot Encoding based on param_solver.json.

    :param genes:
    :param solver:
    :param param_value_dict:
    :param json_param_file:
    :param reverse:
    :return:
    """
    param_names, params = validate_json_file(json_param_file, solver)

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
                params = json_param_file
                originals_ind_to_delete = []
                one_hot_addition = []
                for i, _ in enumerate(param_names):
                    values = []
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
                        one_hot_part = [0 for i in range(max_val - min_val + 1)]
                        one_hot_tail.append(one_hot_part)

        one_hot_tail_len = 0
        for i, _ in enumerate(one_hot_tail):
            one_hot_tail_len += len(one_hot_tail[i])

        one_hot_values = genes[len(genes) - one_hot_tail_len :]

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

        k = 0
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


def get_params_string_from_numeric_params(genes, solver, json_param_file=None):
    """
    Transform string categorical back to string based on params_solver.json.

    :param genes:
    :param solver:
    :param json_param_file:
    :return:
    """
    param_names, params = validate_json_file(json_param_file, solver)

    genes = list(genes)

    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":
            if params[param_names[i]]["valtype"] == "str":
                values = params[param_names[i]]["values"]
                string_value = values[int(genes[i])]
                genes[i] = str(string_value)

    return genes
