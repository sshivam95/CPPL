import json
import os

import jsonschema
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from utils.constants import Constants


def gradient(
    theta: np.ndarray,
    winner_arm: int,
    subset_arms: np.ndarray,
    context_matrix: np.ndarray,
) -> float:
    """
    Calculate the gradient of the log-likelihood function in the partial winner feedback scenario.

    Parameters
    ----------
    theta : Score parameter of each arm in the contender pool.
    winner_arm : Winner arm (parameter) in the subset.
    subset_arms : A subset of arms from the contender pool for solving the instances.
    context_matrix : A context matrix where each element is associated with one of the different arms and contains the
                     properties of the arm itself as well as the context in which the arm needs to be chosen.

    Returns
    -------
    res: The gradient of the log-likelihood function in the partial winner feedback scenario.
    """
    denominator = 0
    num = np.zeros((len(theta)))
    for arm in subset_arms:
        denominator = denominator + np.exp(np.dot(theta, context_matrix[arm, :]))
        num = num + (
            context_matrix[arm, :] * np.exp(np.dot(theta, context_matrix[arm, :]))
        )
    res = context_matrix[winner_arm, :] - (num / denominator)
    return res


def hessian(
    theta: np.ndarray, subset_arms: np.ndarray, context_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate the hessian matrix of the log-likelihood function in the partial winner feedback scenario.

    Parameters
    ----------
    theta : Score parameter of each arm in the contender pool.
    subset_arms : A subset of arms from the contender pool for solving the instances.
    context_matrix : A context matrix where each element is associated with one of the different arms and contains
                     the properties of the arm itself as well as the context in which the arm needs to be chosen.

    Returns
    -------
    A hessian matrix of the log-likelihood function in the partial winner feedback scenario.
    """
    dimension = len(theta)
    t_1 = np.zeros(dimension)
    for arm in subset_arms:
        t_1 = t_1 + (
            context_matrix[arm, :] * np.exp(np.dot(theta, context_matrix[arm, :]))
        )
    num_1 = np.outer(t_1, t_1)
    denominator_1 = 0
    for arm in subset_arms:
        denominator_1 = (
            denominator_1 + np.exp(np.dot(theta, context_matrix[arm, :])) ** 2
        )
    s_1 = num_1 / denominator_1
    num_2 = 0
    for j in subset_arms:
        num_2 = num_2 + (
            np.exp(np.dot(theta, context_matrix[j, :]))
            * np.outer(context_matrix[j, :], context_matrix[j, :])
        )
    denominator_2 = 0
    for arm in subset_arms:
        denominator_2 = denominator_2 + np.exp(np.dot(theta, context_matrix[arm, :]))
    s_2 = num_2 / denominator_2
    return s_1 - s_2


def join_feature_map(x, y, mode: str) -> np.ndarray:
    """
    The feature engineering part of the CPPL algorithm.

    Parameters
    ----------
    x : Features of problem instances.
    y : Features of parameterization.
    mode : Mode of the solver.

    Returns
    -------
    A numpy array of the transforms joint features based on the mode of the solver.
    """
    if mode == "concatenation":
        return np.concatenate((x, y), axis=0)
    elif mode == "kronecker":
        return np.kron(x, y)
    elif mode == "polynomial":
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        return poly.fit_transform(np.concatenate((x, y), axis=0).reshape(1, -1))


def get_problem_instance_list(sorted_directory: dict) -> list:
    """
    Returns clean problem instances as a list.

    Parameters
    ----------
    sorted_directory : A sorted dictionary contains the names of all the problem instances.

    Returns
    -------
    clean_problem_instance_list: A list containing all the names of the problem instances to be solved.
    """
    clean_problem_instance_list = ["" for i, _ in enumerate(sorted_directory)]
    for index, _ in enumerate(sorted_directory):
        clean_problem_instance_list[index] = str(os.fsencode((sorted_directory[index])))
    return clean_problem_instance_list


def json_validation(param: dict, schema: dict) -> bool:
    """
    Validate the json parameters with the schema mentioned.

    Parameters
    ----------
    param : The json type parameters of the solver.
    schema : The meta schema of the json file.

    Returns
    -------
    Boolean whether the given param file validate to the schema.
    """
    try:
        jsonschema.validate(instance=param, schema=schema)
    except jsonschema.exceptions.ValidationError:
        return False
    return True


def set_genes(solver_parameters: dict) -> list:
    """
    Return genes based on solver's parameters.

    Parameters
    ----------
    solver_parameters : The parameters of the solver used to solve the instances.

    Returns
    -------
    genes: List of parameters that are needed.
    """
    param_names = list(solver_parameters.keys())
    genes = [0 for _, _ in enumerate(param_names)]
    for i, _ in enumerate(param_names):
        genes[i] = solver_parameters[param_names[i]]["default"]
    return genes


def get_solver_params(solver_parameters: dict, solver: str) -> (list, dict):
    """
    Return the parameter keys and values from the solver's parameter schema.

    Parameters
    ----------
    solver_parameters : Parameters of the solver.
    solver : Solver used to solve the instances.

    Returns
    -------
    param_names: List of solver's parameters names.
    params: Dictionary data of the solver's parameters.
    """
    if solver_parameters is None:
        json_file_name = "params_" + str(solver)

        with open(
            f"{Constants.PARAMS_JSON_FOLDER.value}/{json_file_name}.json", "r"
        ) as file:
            data = file.read()

        params = json.loads(data)
        param_names = list(params.keys())
    else:
        params = solver_parameters
        param_names = list(params.keys())
    return param_names, params
