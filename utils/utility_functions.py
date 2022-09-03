import os

import jsonschema
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from Configuration_Functions.CPPLConfig import asyncResults


def gradient(theta, Y, S, X):
    denominator = 0
    num = np.zeros((len(theta)))
    for l in S:
        denominator = denominator + np.exp(np.dot(theta,
                                                  X[l, :]))
        num = num + (X[l, :] * np.exp(np.dot(theta,
                                             X[l, :])))
    res = X[Y, :] - (num / denominator)

    return res


def hessian(theta, S, X):
    dimension = len(theta)
    t_1 = np.zeros(dimension)
    for l in S:
        t_1 = t_1 + (X[l, :] * np.exp(np.dot(theta,
                                             X[l, :])))
    num_1 = np.outer(t_1,
                     t_1)
    denominator_1 = 0
    for l in S:
        denominator_1 = denominator_1 + np.exp(np.dot(theta,
                                                      X[l, :])) ** 2
    s_1 = num_1 / denominator_1
    #
    num_2 = 0
    for j in S:
        num_2 = num_2 + (np.exp(np.dot(theta,
                                       X[j, :])) * np.outer(X[j, :],
                                                            X[j, :]))
    denominator_2 = 0
    for l in S:
        denominator_2 = denominator_2 + np.exp(np.dot(theta,
                                                      X[l, :]))
    s_2 = num_2 / denominator_2
    return s_1 - s_2


def join_feature_map(x, y, mode):
    if mode == "concatenation":
        return np.concatenate((x, y),
                              axis=0)
    elif mode == "kronecker":
        return np.kron(x,
                       y)
    elif mode == "polynomial":
        poly = PolynomialFeatures(degree=2,
                                  interaction_only=True)
        return poly.fit_transform(np.concatenate((x, y),
                                                 axis=0).reshape(1,
                                                                 -1))


def get_problem_instance_list(sorted_directory):
    clean_problem_instance_list = ["" for i, _ in enumerate(sorted_directory)]
    for index, _ in enumerate(sorted_directory):
        clean_problem_instance_list[index] = str(
            os.fsencode((sorted_directory[index]))
        )
    return clean_problem_instance_list


def json_validation(param, schema):
    try:
        jsonschema.validate(instance=param,
                            schema=schema)
    except jsonschema.exceptions.ValidationError:
        return False
    return True


def set_genes(solver_parameters: list) -> list:
    param_names = list(solver_parameters.keys())
    genes = [0 for _, _ in enumerate(param_names)]
    for i, _ in enumerate(param_names):
        genes[i] = solver_parameters[param_names[i]]["default"]
    return genes


def save_result(result):
    asyncResults.append([result[0], result[1]])
