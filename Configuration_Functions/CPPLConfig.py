import csv
import json
from multiprocessing import Pool as parallelPool
import os
import os.path
import random

from Configuration_Functions import log_on_huge_params
from Configuration_Functions import pws
from Configuration_Functions import random_genes
from Configuration_Functions import set_param
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


def get_contenders(
        directory,
        filename,
        pca_obj_inst,
        pca_obj_params,
        jfm,
        theta_bar,
        time_step,
        subset_size,
        S_t,
        grad,
        hess_sum,
        grad_op_sum,
        omega,
        solver,
        Pool,
        tracking_Pool,
        min_max_scaler,
        standard_scaler,
        pl,
        param_value_dict,
        json_param_file,
        exp,
):
    X_t, d, features, n, params, v_hat = get_context_feature_matrix(
        Pool,
        directory,
        filename,
        jfm,
        json_param_file,
        min_max_scaler,
        pca_obj_inst,
        pca_obj_params,
        pl,
        solver,
        standard_scaler,
        theta_bar,
    )

    # Todo:
    #  -Create a preselection class with run and initialization of the algorithms.
    discard = []
    if time_step == 0:
        S_t, contender_list, v_hat = preselection_UCB(S_t, X_t, d, grad, grad_op_sum, hess_sum, subset_size, n,
                                                      omega,
                                                      theta_bar, time_step, v_hat)
        if exp == "y":
            for i in range(subset_size):
                contender_list[i] = f"contender_{i}"

        genes = pws.set_genes(json_param_file)
        set_param.set_contender_params("contender_0", genes,
                                       json_param_file)
        contender_list[0] = "contender_0"
        Pool["contender_0"] = genes
        tracking_Pool.info(Pool)

    else:
        # compute confidence_t and select S_t (symmetric group on [num_parameters], consisting of rankings: r ∈ S_n)
        S_t, confidence_t, contender_list, v_hat = preselection_UCB(S_t, X_t, d, grad, grad_op_sum, hess_sum, subset_size, n,
                                                                    omega,
                                                                    theta_bar, time_step, v_hat)

        # update contenders
        for p1 in range(n):
            for p2 in range(n):
                if (
                        p2 != p1
                        and v_hat[p2] - confidence_t[p2] >= v_hat[p1] + confidence_t[p1]
                        and (not p1 in discard)
                ):
                    discard.append(p1)
                    break

        if len(discard) > 0:
            print("There are", len(discard), "parameterizations to discard")
            discard_size = len(discard)

            print(
                "\n *********************************",
                "\n Generating new Parameterizations!",
                "\n *********************************\n",
            )

            # with randomness
            new_candidates_size = 1000

            random_parameters, _ = read_parameters(Pool, solver)

            random_parameters = np.asarray(random_parameters)

            random_parameters = log_on_huge_params.log_space_convert(
                pl, random_parameters, json_param_file
            )

            best_candid = log_on_huge_params.log_space_convert(
                pl,
                random_genes.one_hot_decode(
                    random_parameters[
                        S_t[0],
                    ],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                ),
                json_param_file,
                exp=True,
            )

            second_candid = log_on_huge_params.log_space_convert(
                pl,
                random_genes.one_hot_decode(
                    random_parameters[
                        S_t[1],
                    ],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                ),
                json_param_file,
                exp=True,
            )

            new_candidates_transformed, new_candidates = parallel_evolution_and_fitness(
                subset_size,
                new_candidates_size,
                S_t,
                params,
                solver,
                param_value_dict,
                json_param_file,
                random_genes.genes_set,
                random_genes.one_hot_decode,
                log_on_huge_params.log_space_convert,
                Pool,
                pl,
                best_candid,
                second_candid,
            )

            new_candidates_transformed = min_max_scaler.transform(
                new_candidates_transformed
            )
            new_candidates_transformed = pca_obj_params.transform(
                new_candidates_transformed
            )

            v_hat_new_candidates = np.zeros(new_candidates_size)
            for i in range(new_candidates_size):
                X = join_feature_map(new_candidates_transformed[i], features[0], jfm)
                v_hat_new_candidates[i] = np.exp(np.inner(theta_bar, X))

            best_new_candidates_ind = (-v_hat_new_candidates).argsort()[0:discard_size]

            for i in range(len(best_new_candidates_ind)):
                genes = random_genes.get_params_string_from_numeric_params(
                    log_on_huge_params.log_space_convert(
                        pl,
                        random_genes.one_hot_decode(
                            new_candidates[best_new_candidates_ind[i]],
                            solver,
                            param_value_dict=param_value_dict,
                            json_param_file=json_param_file,
                        ),
                        json_param_file,
                        exp=True,
                    ),
                    solver,
                    json_param_file,
                )

                # Question: What is the use of this check? It is not being used anywhere.
                if type(discard) == list:
                    discard_is_list = True
                else:
                    discard.tolist()

                set_param.set_contender_params(
                    "contender_" + str(discard[i]), genes,
                    json_param_file
                )
                Pool["contender_" + str(discard[i])] = genes
                tracking_Pool.info(Pool)

            contender_list = contender_list_including_generated(
                Pool,
                solver,
                pl,
                params,
                json_param_file,
                theta_bar,
                jfm,
                min_max_scaler,
                pca_obj_params,
                standard_scaler,
                pca_obj_inst,
                directory,
                filename,
                subset_size,
            )

    return X_t, contender_list, discard


def preselection_UCB(S_t, X_t, d, grad, grad_op_sum, hess_sum, subset_size, n, omega, theta_bar, time_step, v_hat):
    if time_step >= 1:
        confidence_t = np.zeros(n)
        hess = hessian(theta_bar, S_t, X_t)
        hess_sum = hess_sum + hess
        grad_op_sum = grad_op_sum + np.outer(grad, grad)
        try:
            V_hat = (1 / time_step) * grad_op_sum

            V_hat = V_hat.astype("float64")

            S_hat = (1 / time_step) * hess_sum

            S_hat = S_hat.astype("float64")

            S_hat_inv = np.linalg.inv(S_hat)

            S_hat_inv = S_hat_inv.astype("float64")

            Sigma_hat = (1 / time_step) * S_hat_inv * V_hat * S_hat_inv  # UCB

            Sigma_hat_sqrt = sp.linalg.sqrtm(Sigma_hat)

            for i in range(n):
                M_i = np.exp(2 * np.dot(theta_bar, X_t[i, :])) * np.dot(
                    X_t[i, :], X_t[i, :]
                )

                confidence_t[i] = omega * np.sqrt(
                    (2 * np.log(time_step) + d + 2 * np.sqrt(d * np.log(time_step)))
                    * np.linalg.norm(Sigma_hat_sqrt * M_i * Sigma_hat_sqrt, ord=2)
                ) # Equation of confidence bound in section 5.3 of https://arxiv.org/pdf/2002.04275.pdf

            S_t = (-(v_hat + confidence_t)).argsort()[0:subset_size]
        except:
            S_t = (-v_hat).argsort()[0:subset_size]

        confidence_t = confidence_t / max(confidence_t)
        v_hat = v_hat / max(v_hat)
        contender_list = update_contender_list(S_t, subset_size)
        return S_t, confidence_t, contender_list, v_hat

    else:
        S_t = (-v_hat).argsort()[0:subset_size]
        contender_list = update_contender_list(S_t, subset_size)
        return S_t, contender_list, v_hat


def update_contender_list(S_t, k):
    contender_list = []
    for i in range(k):
        contender_list.append("contender_" + str(S_t[i]))
    return contender_list


def get_context_feature_matrix(
        Pool,
        directory,
        filename,
        jfm,
        json_param_file,
        min_max_scaler,
        pca_obj_inst,
        pca_obj_params,
        pl,
        solver,
        standard_scaler,
        theta_bar,
):
    # read and preprocess instance features (PCA)
    features = get_features(f"{directory}", f"{filename}")
    features = standard_scaler.transform(features.reshape(1, -1))
    features = pca_obj_inst.transform(features)
    # get parametrization
    params, _ = read_parameters(Pool, solver)
    params = np.asarray(params)
    params_original_size = params.shape[1]
    params = log_on_huge_params.log_space_convert(pl, params, json_param_file)
    params = min_max_scaler.transform(params)
    # PCA on parametrization
    params_transformed = pca_obj_params.transform(params)
    # construct X_t (context specific (instance information) feature matrix ( and parameterization information))
    n = params.shape[0] # Distinct Parameters
    d = len(theta_bar)
    X_t = np.zeros((n, d))
    for i in range(n):
        next_X_t = join_feature_map(
            params_transformed[
                i,
            ],
            features[0],
            jfm,
        )
        X_t[i, :] = next_X_t
    # Normalizing the context specific features
    preprocessing.normalize(X_t, norm="max", copy=False)
    # compute estimated contextualized utility parameters (v_hat)
    v_hat = np.zeros(n)  # Line 7 in CPPL algorithm
    for i in range(n):
        v_hat[i] = np.exp(np.inner(theta_bar, X_t[i, :]))
    return X_t, d, features, n, params, v_hat


asyncResults = []


def contender_list_including_generated(
        Pool,
        solver,
        pl,
        params,
        json_param_file,
        theta_bar,
        jfm,
        min_max_scaler,
        pca_obj_params,
        standard_scaler,
        pca_obj_inst,
        directory,
        filename,
        k,
):
    contender_list = []

    X_t, d, features, n, params, v_hat = get_context_feature_matrix(
        Pool,
        directory,
        filename,
        jfm,
        json_param_file,
        min_max_scaler,
        pca_obj_inst,
        pca_obj_params,
        pl,
        solver,
        standard_scaler,
        theta_bar,
    )

    S_t = (-v_hat).argsort()[0:k]

    for i in range(k):
        contender_list.append("contender_" + str(S_t[i]))

    return contender_list


def save_result(result):
    asyncResults.append([result[0], result[1]])


def parallel_evolution_and_fitness(
        k,
        new_candidates_size,
        S_t,
        params,
        solver,
        param_value_dict,
        json_param_file,
        genes_set,
        One_Hot_decode,
        log_space_convert,
        Pool,
        pl,
        best_candid,
        second_candid,
):
    new_candidates = np.zeros(
        shape=(
            new_candidates_size,
            len(
                random_genes.one_hot_decode(
                    params[
                        S_t[0],
                    ],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                )
            ),
        )
    )
    params_length = len(
        random_genes.one_hot_decode(
            params[
                S_t[0],
            ],
            solver,
            param_value_dict=param_value_dict,
            json_param_file=json_param_file,
        )
    )

    last_step = new_candidates_size % k
    new_candidates_size = new_candidates_size - last_step
    step_size = new_candidates_size / k
    all_steps = []
    for i in range(k):
        all_steps.append(int(step_size))
    all_steps.append(int(last_step))

    step = 0

    pool = parallelPool(processes=k)

    for i in range(len(all_steps)):
        step += all_steps[i]
        pool.apply_async(
            evolution_and_fitness,
            (
                best_candid,
                second_candid,
                len(new_candidates[0]),
                all_steps[i],
                params_length,
                genes_set,
                One_Hot_decode,
                log_space_convert,
                solver,
                json_param_file,
                Pool,
                param_value_dict,
                pl,
            ),
            callback=save_result,
        )

    pool.close()
    pool.join()

    new_candidates_transformed = []
    new_candidates = []

    for i in range(len(asyncResults)):
        for j in range(len(asyncResults[i][0])):
            new_candidates_transformed.append(asyncResults[i][0][j])
            new_candidates.append(asyncResults[i][1][j])

    return new_candidates_transformed, new_candidates


def evolution_and_fitness(
        best_candid,
        second_candid,
        new_candidates,
        new_candidates_size,
        params_length,
        genes_set,
        One_Hot_decode,
        log_space_convert,
        solver,
        json_param_file,
        Pool,
        param_value_dict,
        pl,
):
    # Generation approach based on genetic mechanism with mutation and random individuals
    new_candidates = np.zeros(shape=(new_candidates_size, new_candidates))
    for counter in range(new_candidates_size):
        random_individual = random.uniform(0, 1)
        next_candid = np.zeros(params_length)
        contender = genes_set(solver, json_param_file)
        genes, _ = read_parameters(Pool, solver, contender, json_param_file)
        mutation_genes = random_genes.one_hot_decode(
            genes,
            solver,
            param_value_dict=param_value_dict,
            json_param_file=json_param_file,
        )

        for ii in range(params_length):
            rn = random.uniform(0, 1)
            mutate = random.uniform(0, 1)
            if rn > 0.5:
                next_candid[ii] = best_candid[ii]
            else:
                next_candid[ii] = second_candid[ii]
            if mutate < 0.1:
                next_candid[ii] = mutation_genes[ii]
        if random_individual < 0.99:
            new_candidates[
                counter,
            ] = mutation_genes
        else:
            new_candidates[
                counter,
            ] = next_candid
    new_candidates = random_genes.one_hot_decode(
        new_candidates, solver, json_param_file=json_param_file, reverse=True
    )

    new_candidates_transformed = log_on_huge_params.log_space_convert(
        pl, new_candidates, json_param_file
    )

    return new_candidates_transformed, new_candidates


def update(winner, theta_hat, theta_bar, S_t, X_t, gamma_1, time_step, alpha):
    grad = gradient(theta_hat, winner, S_t, X_t)
    theta_hat = theta_hat + gamma_1 * time_step ** (-alpha) * grad
    theta_hat[theta_hat < 0] = 0
    theta_hat[theta_hat > 1] = 1

    # update theta_bar
    theta_bar = (time_step - 1) * theta_bar / time_step + theta_hat / time_step

    return theta_hat, theta_bar, grad


def get_features(directory, filename):
    with open(f"Instance_Features/Features_{directory}.csv", "r") as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            row_list = row[1:]
            features = [float(j) for j in row_list]
            if os.path.basename(row_list[0]) == filename:
                csvFile.close()
                break
    return np.asarray(features)


def param_read_from_dict(solver, contender, json_param_file, paramNames):
    next_params = contender
    params = json_param_file
    originals_ind_to_delete = []
    one_hot_addition = []
    param_value_dict = {}
    if len(paramNames) == len(contender):
        for i in range(len(paramNames)):
            param_value_dict[paramNames[i]] = next_params[i]
            if params[paramNames[i]]["paramtype"] == "categorical":
                if params[paramNames[i]]["valtype"] == "int":
                    minVal = params[paramNames[i]]["minval"]
                    maxVal = params[paramNames[i]]["maxval"]

                    valuerange = maxVal - minVal + 1

                    values = [minVal + j for j in range(valuerange)]

                    index = int(next_params[i]) - minVal

                elif params[paramNames[i]]["valtype"] == "str":
                    values = params[paramNames[i]]["values"]

                    if type(next_params[i]) == str:
                        index = values.index(next_params[i])
                    else:
                        pass
                if len(values) == 2:
                    # The categorical Parameter can be treated as binary
                    # Replace original value by zero or one
                    for j in range(len(values)):
                        if next_params[i] == values[j]:
                            param_value_dict[paramNames[i]] = j

                elif len(values) > 2:
                    # The categorical Parameter needs One-Hot Encoding
                    # -> append One-Hot Vectors and delete original elements
                    one_hot = [0 for j in range(len(values))]

                    one_hot[index] = 1

                    originals_ind_to_delete.append(i)
                    one_hot_addition += one_hot

                    param_value_dict[paramNames[i]] = None

        new_params = []

        for key in param_value_dict:
            if param_value_dict[key] != None:
                new_params.append(param_value_dict[key])

        new_params += one_hot_addition

    else:
        new_params = contender

    return new_params, param_value_dict


def read_parameters(pool, solver, contender=None, json_param_file=None):
    global parameter_value_dict
    paramNames, params = random_genes.validate_json_file(json_param_file, solver)

    if contender is not None:

        new_params, parameter_value_dict = param_read_from_dict(
            solver, contender, params, paramNames
        )

        return np.asarray(new_params), parameter_value_dict

    elif contender is None:

        if type(pool) != dict:
            Pool_List = list(pool)
            newPool = {}
            for i in range(len(Pool_List)):
                newPool["contender_" + str(i)] = Pool_List[i]
        else:
            newPool = pool

        P = []
        for key in newPool:
            new_params, parameter_value_dict = param_read_from_dict(
                solver, newPool[key], params, paramNames
            )

            P.append(new_params)

        return np.asarray(P), parameter_value_dict


def join_feature_map(x, y, mode):
    if mode == "concatenation":
        return np.concatenate((x, y), axis=0)
    elif mode == "kronecker":
        return np.kron(x, y)
    elif mode == "polynomial":
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        return poly.fit_transform(np.concatenate((x, y), axis=0).reshape(1, -1))


def gradient(theta, Y, S, X):
    denominator = 0
    num = np.zeros((len(theta)))
    for l in S:
        denominator = denominator + np.exp(np.dot(theta, X[l, :]))
        num = num + (X[l, :] * np.exp(np.dot(theta, X[l, :])))
    res = X[Y, :] - (num / denominator)

    return res


def hessian(theta, S, X):
    dimension = len(theta)
    t_1 = np.zeros(dimension)
    for l in S:
        t_1 = t_1 + (X[l, :] * np.exp(np.dot(theta, X[l, :])))
    num_1 = np.outer(t_1, t_1)
    denominator_1 = 0
    for l in S:
        denominator_1 = denominator_1 + np.exp(np.dot(theta, X[l, :])) ** 2
    s_1 = num_1 / denominator_1
    #
    num_2 = 0
    for j in S:
        num_2 = num_2 + (np.exp(np.dot(theta, X[j, :])) * np.outer(X[j, :], X[j, :]))
    denominator_2 = 0
    for l in S:
        denominator_2 = denominator_2 + np.exp(np.dot(theta, X[l, :]))
    s_2 = num_2 / denominator_2
    return s_1 - s_2
