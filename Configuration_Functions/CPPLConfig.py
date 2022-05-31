import numpy as np
import random
import scipy as sp
import os
import re
import csv
import ntpath
import time
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import glob
import os.path
from scipy.linalg import logm, expm
import json
import jsonschema
from jsonschema import validate
from multiprocessing import Pool as parallelPool

from Configuration_Functions import random_genes
from Configuration_Functions import pws
from Configuration_Functions import setParam
from Configuration_Functions import log_on_huge_params


def get_contenders(
    directory,
    filename,
    pca_obj_inst,
    pca_obj_params,
    jfm,
    theta_bar,
    t,
    k,
    Y_t,
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
    # read and preprocess instance features (PCA)
    features = get_features(f"{directory}", f"{filename}")

    features = standard_scaler.transform(features.reshape(1, -1))

    features = pca_obj_inst.transform(features)

    # get parametrization
    params, _ = read_parametrizations(Pool, solver)

    params = np.asarray(params)

    params_original_size = params.shape[1]

    params = log_on_huge_params.log_space_convert(solver, pl, params, json_param_file)

    params = min_max_scaler.transform(params)

    # PCA on parametrization
    params_transformed = pca_obj_params.transform(params)

    # construct X_t (context specific (instance information) feature matrix ( and parameterization information))
    n = params.shape[0]
    d = len(theta_bar)
    X_t = np.zeros((n, d))
    for i in range(n):
        next_X_t = joinFeatureMap(params_transformed[i,], features[0], jfm)
        X_t[i, :] = next_X_t

    # Normalizing the context specific features
    preprocessing.normalize(X_t, norm="max", copy=False)

    # compute estimated contextualized utility parameters (v_hat)
    v_hat = np.zeros(n)
    for i in range(n):
        v_hat[i] = np.exp(np.inner(theta_bar, X_t[i, :]))

    # compute c_t and select S_t (symmetric group on [n], consisting of rankings: r ∈ S_n)
    if t >= 1:
        c_t = np.zeros(n)
        hess = hessian(theta_bar, Y_t, S_t, X_t)
        hess_sum = hess_sum + hess
        grad_op_sum = grad_op_sum + np.outer(grad, grad)
        try:
            V_hat = (1 / t) * grad_op_sum

            V_hat = V_hat.astype("float64")

            S_hat = (1 / t) * hess_sum

            S_hat = S_hat.astype("float64")

            S_hat_inv = np.linalg.inv(S_hat)

            S_hat_inv = S_hat_inv.astype("float64")

            Sigma_hat = (1 / t) * S_hat_inv * V_hat * S_hat_inv

            Sigma_hat_sqrt = sp.linalg.sqrtm(Sigma_hat)

            for i in range(n):
                M_i = np.exp(2 * np.dot(theta_bar, X_t[i, :])) * np.dot(
                    X_t[i, :], X_t[i, :]
                )

                c_t[i] = omega * np.sqrt(
                    (2 * np.log(t) + d + 2 * np.sqrt(d * np.log(t)))
                    * np.linalg.norm(Sigma_hat_sqrt * M_i * Sigma_hat_sqrt, ord=2)
                )

            S_t = (-(v_hat + c_t)).argsort()[0:k]

        except:

            S_t = (-(v_hat)).argsort()[0:k]
            pass
    else:
        S_t = (-(v_hat)).argsort()[0:k]

    if t >= 1:
        c_t = c_t / max(c_t)
        v_hat = v_hat / max(v_hat)
        # print('\nc_t\n',c_t,'\n')
        # print('Index highest c_t =',c_t.argmax(axis=0),'\n')

    contender_list = []
    for i in range(k):
        contender_list.append("contender_" + str(S_t[i]))

    # print('\nV_hat\n',v_hat,'\n')
    # print('Index highest V_hat =',v_hat.argmax(axis=0),'\n')

    discard = []

    # update contenders
    if t >= 1:
        for p1 in range(n):
            for p2 in range(n):
                if (
                    p2 != p1
                    and v_hat[p2] - c_t[p2] >= v_hat[p1] + c_t[p1]
                    and (not (p1 in discard))
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

            ## with randomness
            new_candids_size = 1000

            global asyncResults
            asyncResults = []

            parametrizations, _ = read_parametrizations(Pool, solver)

            parametrizations = np.asarray(parametrizations)

            parametrizations = log_on_huge_params.log_space_convert(
                solver, pl, parametrizations, json_param_file
            )

            best_candid = log_on_huge_params.log_space_convert(
                solver,
                pl,
                random_genes.One_Hot_decode(
                    parametrizations[S_t[0],],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                ),
                json_param_file,
                exp=True,
            )

            second_candid = log_on_huge_params.log_space_convert(
                solver,
                pl,
                random_genes.One_Hot_decode(
                    parametrizations[S_t[1],],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                ),
                json_param_file,
                exp=True,
            )

            new_candids_transformed, new_candids = parallel_evolution_and_fitness(
                k,
                new_candids_size,
                S_t,
                params,
                solver,
                param_value_dict,
                json_param_file,
                random_genes.genes_set,
                random_genes.One_Hot_decode,
                log_on_huge_params.log_space_convert,
                Pool,
                pl,
                best_candid,
                second_candid,
            )

            new_candids_transformed = min_max_scaler.transform(new_candids_transformed)
            new_candids_transformed = pca_obj_params.transform(new_candids_transformed)

            v_hat_new_candids = np.zeros(new_candids_size)
            for i in range(new_candids_size):
                X = joinFeatureMap(new_candids_transformed[i], features[0], jfm)
                v_hat_new_candids[i] = np.exp(np.inner(theta_bar, X))

            best_new_candids_ind = (-v_hat_new_candids).argsort()[0:discard_size]

            for i in range(len(best_new_candids_ind)):
                genes = random_genes.getParamsStringFromNumericParams(
                    log_on_huge_params.log_space_convert(
                        solver,
                        pl,
                        random_genes.One_Hot_decode(
                            new_candids[best_new_candids_ind[i]],
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

                if type(discard) == list:
                    discard_is_list = True
                else:
                    discard.tolist()

                setParam.set_params(
                    "contender_" + str(discard[i]), genes, solver, json_param_file
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
                k,
            )

    if t == 0:

        if exp == "y":
            for i in range(k):
                contender_list[i] = f"contender_{i}"

        genes = pws.set_genes(solver, json_param_file)
        setParam.set_params("contender_0", genes, solver, json_param_file)
        contender_list[0] = "contender_0"
        Pool["contender_0"] = genes
        tracking_Pool.info(Pool)

    return X_t, contender_list, discard


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
    # read and preprocess instance features (PCA)
    features = get_features(f"{directory}", f"{filename}")

    features = standard_scaler.transform(features.reshape(1, -1))

    features = pca_obj_inst.transform(features)

    # get parametrizations
    params, _ = read_parametrizations(Pool, solver)

    params = np.asarray(params)

    params_original_size = params.shape[1]

    params = log_on_huge_params.log_space_convert(solver, pl, params, json_param_file)

    params = min_max_scaler.transform(params)

    # PCA on parametrizations
    params_transformed = pca_obj_params.transform(params)

    # construct X_t (context specific (instance information) feature matrix (parameterizations information))
    n = params.shape[0]
    d = len(theta_bar)
    X_t = np.zeros((n, d))

    contender_list = []

    for i in range(n):
        next_X_t = joinFeatureMap(params_transformed[i,], features[0], jfm)
        X_t[i, :] = next_X_t

    # Normalizing the context specific features
    preprocessing.normalize(X_t, norm="max", copy=False)

    v_hat = np.zeros(n)

    for i in range(n):
        v_hat[i] = np.exp(np.inner(theta_bar, X_t[i, :]))

    S_t = (-v_hat).argsort()[0:k]

    for i in range(k):
        contender_list.append("contender_" + str(S_t[i]))

    return contender_list


def saveResult(result):
    asyncResults.append([result[0], result[1]])


def parallel_evolution_and_fitness(
    k,
    new_candids_size,
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
    new_candids = np.zeros(
        shape=(
            new_candids_size,
            len(
                random_genes.One_Hot_decode(
                    params[S_t[0],],
                    solver,
                    param_value_dict=param_value_dict,
                    json_param_file=json_param_file,
                )
            ),
        )
    )
    params_length = len(
        random_genes.One_Hot_decode(
            params[S_t[0],],
            solver,
            param_value_dict=param_value_dict,
            json_param_file=json_param_file,
        )
    )

    last_step = new_candids_size % k
    new_candids_size = new_candids_size - last_step
    step_size = new_candids_size / k
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
                len(new_candids[0]),
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
            callback=saveResult,
        )

    pool.close()
    pool.join()

    new_candids_transformed = []
    new_candids = []

    for i in range(len(asyncResults)):
        for j in range(len(asyncResults[i][0])):
            new_candids_transformed.append(asyncResults[i][0][j])
            new_candids.append(asyncResults[i][1][j])

    return new_candids_transformed, new_candids


def evolution_and_fitness(
    best_candid,
    second_candid,
    new_candids,
    new_candids_size,
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
    new_candids = np.zeros(shape=(new_candids_size, new_candids))
    for counter in range(new_candids_size):
        random_individual = random.uniform(0, 1)
        next_candid = np.zeros(params_length)
        contender = genes_set(solver, json_param_file)
        genes, _ = read_parametrizations(Pool, solver, contender, json_param_file)
        mutation_genes = random_genes.One_Hot_decode(
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
            new_candids[counter,] = mutation_genes
        else:
            new_candids[counter,] = next_candid
    new_candids = random_genes.One_Hot_decode(
        new_candids, solver, json_param_file=json_param_file, reverse=True
    )

    new_candids_transformed = log_on_huge_params.log_space_convert(
        solver, pl, new_candids, json_param_file
    )

    return new_candids_transformed, new_candids


def update(winner, theta_hat, theta_bar, S_t, X_t, gamma_1, t, alpha):
    grad = gradient(theta_hat, winner, S_t, X_t)
    theta_hat = theta_hat + gamma_1 * t ** (-alpha) * grad
    theta_hat[theta_hat < 0] = 0
    theta_hat[theta_hat > 1] = 1

    # update theta_bar
    theta_bar = (t - 1) * theta_bar / t + theta_hat / t

    return theta_hat, theta_bar, grad


def get_features(directory, filename):
    with open(f"Instance_Features/Features_{directory}_5000.csv", "r") as csvFile:
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


def read_parametrizations(pool, solver, contender=None, json_param_file=None):
    if json_param_file == None:
        json_file_name = "params_" + str(solver)

        with open(f"Configuration_Functions/{json_file_name}.json", "r") as f:
            data = f.read()
        params = json.loads(data)

        paramNames = list(params.keys())

    else:
        params = json_param_file
        paramNames = list(params.keys())

    if contender != None:

        new_params, param_value_dict = param_read_from_dict(
            solver, contender, params, paramNames
        )

        return np.asarray(new_params), param_value_dict

    elif contender == None:

        if type(pool) != dict:
            Pool_List = list(pool)
            newPool = {}
            for i in range(len(Pool_List)):
                newPool["contender_" + str(i)] = Pool_List[i]
        else:
            newPool = pool

        P = []
        for key in newPool:
            new_params, param_value_dict = param_read_from_dict(
                solver, newPool[key], params, paramNames
            )

            P.append(new_params)

        return np.asarray(P), param_value_dict


def joinFeatureMap(x, y, mode):
    if mode == "concatenation":
        return np.concatenate((x, y), axis=0)
    elif mode == "kronecker":
        return np.kron(x, y)
    elif mode == "polynomial":
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        return poly.fit_transform(np.concatenate((x, y), axis=0).reshape(1, -1))


def gradient(theta, Y, S, X):
    denom = 0
    num = np.zeros((len(theta)))
    for l in S:
        denom = denom + np.exp(np.dot(theta, X[l, :]))
        num = num + (X[l, :] * np.exp(np.dot(theta, X[l, :])))
    res = X[Y, :] - (num / denom)

    return res


def hessian(theta, Y, S, X):
    d = len(theta)
    t_1 = np.zeros((d))
    for l in S:
        t_1 = t_1 + (X[l, :] * np.exp(np.dot(theta, X[l, :])))
    num_1 = np.outer(t_1, t_1)
    denom_1 = 0
    for l in S:
        denom_1 = denom_1 + np.exp(np.dot(theta, X[l, :])) ** 2
    s_1 = num_1 / denom_1
    #
    num_2 = 0
    for j in S:
        num_2 = num_2 + (np.exp(np.dot(theta, X[j, :])) * np.outer(X[j, :], X[j, :]))
    denom_2 = 0
    for l in S:
        denom_2 = denom_2 + np.exp(np.dot(theta, X[l, :]))
    s_2 = num_2 / denom_2
    return s_1 - s_2
