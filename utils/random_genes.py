"""Random Genes Utils."""
import math
import random
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.random import choice

from utils.utility_functions import get_solver_params


def get_all_min_and_max(solver_parameters: Dict) -> Union[List[int], List[int]]:
    """
    Get the minimum and maximum values from the category of parameters of the solver.

    Parameters
    ----------
    solver_parameters : Dict
        The parameters of solver from the json file.

    Returns
    -------
    all_min: List[int]
        All minimum values from the file.
    all_max: List[int]
        All maximum values from the file.
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


def get_genes_set(solver: str, solver_parameters: Dict = None) -> List[int]:
    """
    Return the gene set (parameter set) for a particular solver.


    Parameters
    ----------
    solver : str
        The solver name for which the gene set is required.
    solver_parameters : Dict, default=None
        The parameters of solver from the json file.

    Returns
    -------
    genes: List[int]
        The generated genes set.
    """
    param_names, params = get_solver_params(
        solver_parameters=solver_parameters, solver=solver
    )

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

    return genes

def get_log_distribution_params(
    default: int,
    parameter_index: int,
    max_val: int,
    min_val: int,
    param_names: List,
    params: Dict,
    splittable: bool,
) -> Tuple[float, bool, float, bool, bool, float, float, float, List[float]]:
    """
    Return the parameters in the logarithm space.

    Parameters
    ----------
    default : int
        The default value of the parameter.
    parameter_index : int
        The parameter index.
    max_val : int
        Value of the `maxval` property of the solver's parameter at parameter index.
    min_val : int
        Value of the `minval` property of the solver's parameter at parameter index.
    param_names : List
        Parameter name of the solver's parameter at parameter index.
    params : Dict
        Solver's parameter set.
    splittable : bool
        Boolean value of the parameter's `splitbydefault` property.

    Returns
    -------
    high: float
        The Highest exponential value based on a random uniform distribution between with upper bound as
        `maxval` or log of `maxval` property (if `logonneg` is true) of solver parameter.
    include_zero: bool
        Boolean value of the `includezero` property of solver parameter.
    log_max_val: float
        Logarithm of the value of `maxval` property of the solver's parameter at parameter index.
    log_on_neg: bool
        Boolean value of the `logonneg` property of solver parameter.
    log_on_pos: bool
        Boolean value of the `logonpos` property of solver parameter.
    low: float
        The Lowest exponential value based on a random uniform distribution between with lower bound as
        `minval` or log of `minval` property (if `logonneg` is true) of solver parameter.
    probability_positive: float
        Value of the `probabpos` or `probabneg` property (whichever is available) of the
        solver parameter.
    probability_zero: float
        Value of the `probabilityzero` property (if available else 0.1) of the solver parameter.
    weights: List[float]
        Probability weights of the parameter at the given index.
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

    if "logonpos" in params[param_names[parameter_index]]:
        if params[param_names[parameter_index]]["logonpos"]:
            log_on_pos = True
        if "probabpos" in params[param_names[parameter_index]]:
            probability_positive = params[param_names[parameter_index]]["probabpos"]
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
        if "logonneg" in params[param_names[parameter_index]]:
            if params[param_names[parameter_index]]["logonneg"]:
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


def split_by_default(
    index: int, param_names: List, params: List
) -> Tuple[int, int, int, bool]:
    """
    Check if the solver parameters have splittable by default property or not.

    Parameters
    ----------
    index : int
        Index of the solver parameter.
    param_names : List
        Name of the solver parameter.
    params : List
        Parameter set of the solver.

    Returns
    -------
    default: int
        The default value of the parameter.
    max_val: int
        Max value of the parameter.
    min_val: int
        Min value of the parameter.
    splittable: bool
        Boolean value whether the parameter `splitbydefault` is set to true or false
    """
    max_val = params[param_names[index]]["maxval"]
    min_val = params[param_names[index]]["minval"]
    default = None
    splittable = False
    if "splitbydefault" in params[param_names[index]]:
        if params[param_names[index]]["splitbydefault"]:
            default = params[param_names[index]]["default"]
            splittable = True
            return default, max_val, min_val, splittable
    return default, max_val, min_val, splittable


def get_one_hot_decoded_param_set(
    genes: Union[List, np.ndarray],
    solver: str,
    param_value_dict: Dict = None,
    solver_parameters: Dict = None,
    reverse: bool = False,
) -> List:
    """
    Decoding the parameters and returning new parameters or parameters like a one-hot vector back to solver specific
    representation based on param_solver.json

    Parameters
    ----------
    genes : Union[List, np.ndarray]
        Genes or parameters (can be random) to be added in the paarameter pool
    solver : str
        name of the solver used to solve problem instances.
    param_value_dict : Dict, default=None
        Dictionary of contenders parameters.
    solver_parameters : Dict, default=None
        Dictionary of solver's parameters.
    reverse : bool, default=False
        Boolean value whether the parameters are in reverse order or not.

    Returns
    -------
    List
        New or old set or parameters.
    """
    param_names, params = get_solver_params(
        solver_parameters=solver_parameters, solver=solver
    )

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


def get_params_string_from_numeric_params(
    genes: np.ndarray, solver: str, solver_parameters: Dict = None
) -> List[str]:
    """
    Transform string categorical back to string based on solver parameters file `params_{solver}.json.`

    Parameters
    ----------
    genes : np.ndarry
        Parameter set of the contender.
    solver : str
        Solver used to solve problem instances.
    solver_parameters : Dict, default=None
        The parameters used by solver to solve problem instances.

    Returns
    -------
    genes: List[str]
        List of parameters in form of string if the solver parameter type is categorical.
    """
    param_names, params = get_solver_params(
        solver_parameters=solver_parameters, solver=solver
    )

    genes = list(genes)

    for i, _ in enumerate(param_names):
        if params[param_names[i]]["paramtype"] == "categorical":
            if params[param_names[i]]["valtype"] == "str":
                values = params[param_names[i]]["values"]
                string_value = values[int(genes[i])]
                genes[i] = str(string_value)

    return genes
