# Start ReACTR_CPPL by:
# python3 ReACTR_CPPL.py -d modular_kcnf...
# ... -to 300 -p pws -s cadical

import argparse
import csv
import json
import multiprocessing as mp
import os
import os.path
import pathlib
import signal
import sys
import time
from multiprocessing import Manager

import jsonschema
import numpy as np
from jsonschema import validate
from sklearn import preprocessing
from sklearn.decomposition import PCA

import utils.log_params_utils
from Configuration_Functions import CPPLConfig
from Configuration_Functions import file_logging
from Configuration_Functions import log_on_huge_params
from Configuration_Functions import pws
from Configuration_Functions import random_genes
from Configuration_Functions import set_param
from Configuration_Functions import tournament as solving
from Configuration_Functions.random_genes import genes_set


def _main():
    # global args, solver, directory, files, times_instances, problem_instance_list, tracking_times, tracking_pool, num_parameters, solver_parameters, contender_pool, f, rounds_to_train, standard_scaler, pca_obj_inst, params, parameter_value_dict, min_max_scaler, pca_obj_params, jfm, theta_hat, theta_bar, grad_op_sum, hess_sum, omega, gamma_1, alpha, t, winner_index_time_step, S_t, grad, winner_known, dimensions
    global training_files
    # Parse directory of instances, solver, max. time_step for single solving
    parser = argparse.ArgumentParser(description="Start Tournaments")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="No Problem Instance Directory given",
        help="Directory for instances",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="No solver chosen",
        help="Solver/math. model as .py",
    )
    parser.add_argument(
        "-to",
        "--timeout",
        type=int,
        default=300,
        help="""Stop solving single instance after (int) seconds [300]""",
    )
    parser.add_argument(
        "-nc",
        "--contenders",
        type=int,
        default=30,
        help="The number of contenders [30]",
    )
    parser.add_argument(
        "-keeptop",
        "--top_to_keep",
        type=int,
        default=2,
        help="""Number of top contenders to get chosen for game [2]""",
    )
    parser.add_argument(
        "-p", "--pws", type=str, default=None, help="Custom Parameter Genome []"
    )
    parser.add_argument(
        "-usedata",
        "--data",
        type=str,
        default=None,
        help="""Type y if prior gene and score data should be used []""",
    )
    parser.add_argument(
        "-experimental",
        "--exp",
        type=str,
        default=None,
        help="""Type y if prior gene and score data should
                    be experimental (Pool_exp.txt) []""",
    )
    parser.add_argument(
        "-ch",
        "--chance",
        type=int,
        default=25,
        help="""Chance to replace gene randomly in percent (int: 0 - 100) [25]""",
    )
    parser.add_argument(
        "-m",
        "--mutate",
        type=int,
        default=10,
        help="""Chance for mutation in crossover process in percent (int: 0 - 100) [10]""",
    )
    parser.add_argument(
        "-k",
        "--kill",
        type=float,
        default=5,
        help="""Contenders with a variance higher than this are killed and replaced (float) [5]""",
    )
    parser.add_argument(
        "-tn",
        "--train_number",
        type=float,
        default=None,
        help="""How many of the first instances are to be trained on before starting (int) [None] """,
    )
    parser.add_argument(
        "-tr",
        "--train_rounds",
        type=float,
        default=0,
        help="""How many rounds are the first -tn instances to be trained on (int) [1] """,
    )
    parser.add_argument(
        "-fo",
        "--file_order",
        type=str,
        default="ascending",
        help="""Specify the order by which the problem instances are solved""",
    )
    parser.add_argument(
        "-nc_pca_f",
        "--nc_pca_f",
        type=int,
        default=7,
        help="""Number of the dimensions for the PCA of the instance features """,
    )
    parser.add_argument(
        "-nc_pca_p",
        "--nc_pca_p",
        type=int,
        default=10,
        help="""Number of the dimensions for the PCA of the parameter (features) """,
    )
    parser.add_argument(
        "-jfm",
        "--jfm",
        type=str,
        default="polynomial",
        help="""Mode of the joined feature map""",
    )
    parser.add_argument(
        "-omega",
        "--omega",
        type=float,
        default=0.001,
        help="""Omega parameter for CPPL""",
    )
    parser.add_argument(
        "-gamma", "--gamma", type=float, default=1, help="""Gamma parameter for CPPL"""
    )
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        default=0.2,
        help="""Alpha parameter for CPPL""",
    )
    parser.add_argument(
        "-tfn",
        "--times_file_name",
        type=str,
        default="Times_per_instance_CPPL.txt",
        help="""Name of the file which the times needed to solve instances are tracked in""",
    )
    parser.add_argument(
        "-pl",
        "--paramlimit",
        type=float,
        default=100000,
        help="""Limit for the possible absolute value of
                    a parameter for it to be normed to log space before CPPL computation""",
    )
    parser.add_argument(
        "-bp",
        "--baselineperf",
        type=bool,
        default=False,
        help="""Set to true if only default parameterization should run""",
    )
    args, unknown = parser.parse_known_args()
    solver = args.solver
    original_chance = (
        args.chance
    )  # Question: What are the uses of these? They are not used in the code anywhere, is it safe to delete them?
    original_mutation = args.mutate
    original_timeout = args.timeout

    (
        directory,
        times_instances,
        problem_instance_list,
    ) = _init_output(args=args)

    # Checking Solver
    if args.solver == "No solver chosen":
        print("\nYou need to choose a solver!!!\n\n**[-s <solver_name>]**\n\n")
        sys.exit(0)

    # Creating tracking logs
    tracking_times = file_logging.tracking_files(
        filename=args.times_file_name, logger_name="ReACTR_CPPL", level="INFO"
    )
    tracking_pool = file_logging.tracking_files(
        filename="Pool.txt", logger_name="ReACTR_Pool", level="INFO"
    )

    # Count available cores
    num_parameters = mp.cpu_count()  # Total number of Parameters in set P
    _init_parameter_directory()
    solver_parameters = validate_param_json(solver=solver)

    contender_pool = _init_pool(
        args=args, solver_parameters=solver_parameters, solver=solver
    )

    # Write contender_pool to textfile for solver to access parameter settings
    tracking_pool.info(contender_pool)
    # If training is required, prepare
    if args.train_number is not None:
        for root, training_directory, training_files in sorted(os.walk(directory)):
            continue
    else:
        training_files = None
    rounds_to_train = int(args.train_rounds)

    ###################################################################
    _check_instance_feature_directory(
        args=args, directory=directory
    )  # Check if the instance feature directory available

    directory, features, standard_scaler, train_list = _init_features(
        directory=directory
    )  # initialize features from instance files

    no_comp_pca_features, pca_obj_inst = _init_pca_features(
        args=args, features=features
    )  # initialize pca features

    # Get parameters and apply PCA
    (
        min_max_scaler,
        num_pca_params_components,
        parameter_value_dict,
        params,
        pca_obj_params,
    ) = _get_pca_params(
        args=args,
        contender_pool=contender_pool,
        solver_parameters=solver_parameters,
        solver=solver,
    )

    # other parameters
    (
        S_t,
        Y_t,
        alpha,
        gamma_1,
        grad,
        grad_op_sum,
        hess_sum,
        jfm,
        omega,
        time_step,
        theta_bar,
        theta_hat,
    ) = _get_other_params(args, no_comp_pca_features, num_pca_params_components)
    #########################################
    winner_known = True
    return (
        args,
        solver,
        directory,
        times_instances,
        problem_instance_list,
        tracking_times,
        tracking_pool,
        num_parameters,
        solver_parameters,
        contender_pool,
        training_files,  # f in main code
        train_list,
        rounds_to_train,
        standard_scaler,
        pca_obj_inst,
        params,
        parameter_value_dict,
        min_max_scaler,
        pca_obj_params,
        jfm,
        theta_hat,
        theta_bar,
        grad_op_sum,
        hess_sum,
        omega,
        gamma_1,
        alpha,
        time_step,
        Y_t,
        S_t,
        grad,
        winner_known,
    )


# Todo: Create an initialization class to call all the initialization methods
def _check_instance_feature_directory(args, directory):
    instance_feature_directory = pathlib.Path("Instance_Features")
    if instance_feature_directory.exists():
        pass
    else:
        print(
            "\nWARNING!\n\nA directory named <Instance_Features> with a "
            ".csv file containing the instance features is necessary!"
        )
        if args.directory != "No Problem Instance Directory given":
            print("\nIt must be named: Features_" + str(directory)[2:-1] + ".csv")
        else:
            print(
                "\nIt must be named: " "Features_<problem_instance_directory_name>.csv"
            )
        print(
            "\ninstance1 feature_value1 feature_value2 "
            "....\ninstance2 feature_value1..."
        )
        print("\nExiting...")
        sys.exit(0)


def _get_other_params(args, no_comp_pca_features, num_pca_params_components):
    jfm = args.jfm  # 'polynomial'
    dimensions = 4  # by default # Corresponds to n different parameterizations (Pool of candidates)
    if jfm == "concatenation":
        dimensions = no_comp_pca_features + num_pca_params_components
    elif jfm == "kronecker":
        dimensions = no_comp_pca_features * num_pca_params_components
    elif jfm == "polynomial":
        for index_pca_params in range(
            (no_comp_pca_features + num_pca_params_components) - 2
        ):
            dimensions = dimensions + 3 + index_pca_params
    # theta_hat = np.random.rand(dimensions)
    theta_hat = np.zeros(dimensions)  # Line 2 CPPL (random Parameter Vector)
    theta_bar = theta_hat  # Line 3 CPPl
    grad_op_sum = np.zeros((dimensions, dimensions))
    hess_sum = np.zeros((dimensions, dimensions))
    omega = (
        args.omega
    )  # Parameter of CPPL *Helps determine the confidence intervals (Best value = 0.001)
    gamma_1 = args.gamma  # Parameter CPPL (Best value = 1)
    alpha = args.alpha  # Parameter CPPL (Best value = 0.2)
    time_step = 0  # initial time step = 0 where initialization takes place
    Y_t = 0
    S_t = []  # Subset of contenders Line 9 of CPPL
    grad = np.zeros(dimensions)  # Gradient ∇L in line 11 to update ˆθt
    return (
        S_t,
        Y_t,
        alpha,
        gamma_1,
        grad,
        grad_op_sum,
        hess_sum,
        jfm,
        omega,
        time_step,
        theta_bar,
        theta_hat,
    )


def _get_pca_params(args, contender_pool, solver_parameters, solver):
    params, parameter_value_dict = CPPLConfig.read_parameters(contender_pool, solver)
    params = np.asarray(params)
    all_min, all_max = random_genes.get_all_min_and_max(solver_parameters)
    all_min, _ = CPPLConfig.read_parameters(contender_pool, solver, contender=all_min)
    all_max, _ = CPPLConfig.read_parameters(contender_pool, solver, contender=all_max)
    params = np.append(params, [all_min], axis=0)
    params = np.append(params, [all_max], axis=0)
    params = utils.log_params_utils.log_space_convert(
        float(args.paramlimit), params, solver_parameters
    )
    min_max_scaler = preprocessing.MinMaxScaler()
    params = min_max_scaler.fit_transform(params)
    num_pca_params_components = args.nc_pca_p
    pca_obj_params = PCA(n_components=num_pca_params_components)
    pca_obj_params.fit(params)
    return (
        min_max_scaler,
        num_pca_params_components,
        parameter_value_dict,
        params,
        pca_obj_params,
    )


def _init_pca_features(args, features):
    # PCA on features
    no_comp_pca_features = args.nc_pca_f
    pca_obj_inst = PCA(n_components=no_comp_pca_features)
    pca_obj_inst.fit(features)
    return no_comp_pca_features, pca_obj_inst


def _init_features(directory):
    # read features
    if os.path.isfile(
        "Instance_Features/training_features_" + str(directory)[2:-1] + ".csv"
    ):
        pass
    else:
        print(
            "\n\nThere needs to be a file with training instance features "
            "named << training_features_" + str(directory)[2:-1] + ".csv >> in"
            " the directory Instance_Features\n\n"
        )
        sys.exit(0)
    features = []
    train_list = []
    directory = str(directory)[2:-1]
    with open(f"Instance_Features/training_features_{directory}.csv", "r") as csvFile:  # Question: If the training functionality was not pursued, can we comment this out as well?
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            if len(row[0]) != 0:
                next_features = row
                train_list.append(row[0])  # Question: What's the use of this?
                next_features.pop(0)
                next_features = [float(j) for j in next_features]
                features.append(next_features)
    csvFile.close()
    features = np.asarray(features)
    standard_scaler = preprocessing.StandardScaler()
    features = standard_scaler.fit_transform(features)
    return directory, features, standard_scaler, train_list


def _init_parameter_directory():
    parameter_directory = pathlib.Path("ParamPool")
    if parameter_directory.exists():
        pass
    else:
        os.mkdir("ParamPool")


def _init_output(args):
    # global directory, files, times_instances, problem_instance_list, tracking_times, tracking_pool
    # Initialize output of times needed for solving one instance
    if args.directory != "No Problem Instance Directory given":
        directory = os.fsencode(args.directory)
        path, dirs, files = next(
            os.walk(args.directory)
        )  # Question: path, dirs and files are not used anywhere, what is its purpose?
        file_count = len(files)  # Question: It is not used anywhere, safe to delete?
        times_instances = []
        if args.file_order == "ascending":
            problem_instance_list = _init_problem_instance_list(
                sorted(os.listdir(directory))
            )
        elif args.file_order == "descending":
            problem_instance_list = _init_problem_instance_list(
                sorted(os.listdir(directory), reverse=True)
            )
        else:
            file_order = str(args.file_order)
            with open(f"{file_order}.txt", "r") as file:
                problem_instance_list = eval(file.read())
        with open("problem_instance_list.txt", "w") as file:
            print(problem_instance_list, file=file)
    else:
        print(
            "\n\nYou need to specify a directory containing the problem instances!\n\n**[-d directory_name]**\n\n"
        )
        sys.exit(0)

    return directory, times_instances, problem_instance_list


def _init_problem_instance_list(problem_instance_list):
    clean_problem_instance_list = ["" for _ in range(len(problem_instance_list))]
    for index in range(len(problem_instance_list)):
        clean_problem_instance_list[index] = str(
            os.fsdecode(problem_instance_list[index])
        )
    problem_instance_list = clean_problem_instance_list
    return problem_instance_list


def _init_pool(args, solver_parameters, solver):
    # global contender_pool
    # Initialize contender_pool
    if args.data is None:
        pool_keys = ["contender_{0}".format(c) for c in range(args.contenders)]
        contender_pool = dict.fromkeys(pool_keys, 0)
        if args.baselineperf:
            print("Baseline Performance Run (only default parameters)")
            for key in contender_pool:
                contender_pool[key] = pws.set_genes(solver_parameters)
                set_param.set_contender_params(
                    key, contender_pool[key], solver_parameters
                )
        else:
            for key in contender_pool:
                contender_pool[key] = genes_set(solver)
                set_param.set_contender_params(
                    key, contender_pool[key], solver_parameters
                )
            if args.pws is not None:
                contender_pool["contender_0"] = pws.set_genes(solver_parameters)
                set_param.set_contender_params(
                    "contender_0",
                    contender_pool["contender_0"],
                    solver_parameters,
                )

    elif args.data == "y":
        pool_file = "Pool.txt"
        if args.exp is None:
            pool_file = "Pool.txt"
        elif args.exp == "y":
            pool_file = f"Pool_exp_{solver}.txt"
        with open(f"{pool_file}", "r") as file:
            contender_pool = eval(file.read())
            for key in contender_pool:
                set_param.set_contender_params(
                    contender_index=key, genes=contender_pool[key], solver_parameters=solver_parameters
                )

    return contender_pool


def _init_data_structures():
    # Initialize pickled data
    multiprocessing_event = mp.Event()
    mp.freeze_support()
    event = Manager().list([0])
    winner = Manager().list([None])
    res = Manager().list([[0, 0] for _ in range(num_parameters)])
    interim = mp.Manager().list([0 for _ in range(num_parameters)])
    if solver == "cadical":
        interim = Manager().list([[0, 0, 0, 0, 0, 0] for _ in range(num_parameters)])
    elif solver == "glucose":
        interim = Manager().list(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(num_parameters)]
        )
    elif solver == "cplex":
        interim = Manager().list([[1000000, 100, 0, 0] for _ in range(num_parameters)])
    new_time = Manager().list([args.timeout])
    process_ids = Manager().list([[0] for _ in range(num_parameters)])
    sub_start = Manager().list([[0] for _ in range(num_parameters)])

    # Initialize parallel solving data
    process = ["process_{0}".format(s) for s in range(num_parameters)]
    results = [[0 for s in range(2)] for c in range(num_parameters)]
    interim_res = [[0 for s in range(3)] for c in range(num_parameters)]
    start = time.time()
    winner_known = True

    return (
        multiprocessing_event,
        event,
        winner,
        res,
        interim,
        new_time,
        process_ids,
        sub_start,
        process,
        results,
        interim_res,
        start,
        winner_known,
    )


# Ordering the results with this
def find_best(list1, number):
    list_new = sorted(list1.items(), key=lambda kv: kv[1][1])
    return list_new[:number]


# Todo: create different class to validate json
def validate_param_json(solver):
    json_file_name = "params_" + str(solver)

    with open(f"Configuration_Functions/{json_file_name}.json", "r") as f:
        data = f.read()
    params = json.loads(data)

    param_names = list(params.keys())

    with open("Configuration_Functions/paramSchema.json", "r") as f:
        schema = f.read()
    schemata = json.loads(schema)

    def json_validation(jsonfile):
        try:
            validate(instance=jsonfile, schema=schemata)
        except jsonschema.exceptions.ValidationError as err:
            return False
        return True

    for pn in param_names:
        valid = json_validation(params[pn])
        if not valid:
            print(params[pn])
            print("Invalid JSON data structure. Exiting.")
            sys.exit(0)

    return params


# Todo: Create a different class or tournament runs
def non_nlock_read(output):
    # fd = output.fileno()
    # fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    # fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        return output.readline()
    except:
        return ""


def start_run(
    args,
    filename,
    timelimit,
    params,
    core,
    sub_start,
    pids,
    results,
    interim,
    ev,
    event,
    new_time,
):
    sub_start[core] = time.process_time()
    if args.baselineperf:
        params = []
    else:
        pass
    proc = solving.start(params, timelimit, filename, solver)
    pid = proc.pid
    pids[core] = pid
    awaiting = True
    while awaiting:

        line = non_nlock_read(proc.stdout)

        if line != b"":
            output = solving.check_output(line, interim[core], solver)
            if output != "No output" and output is not None:
                interim[core] = output

            if_solved = solving.check_if_solved(
                line, results[core], proc, event, non_nlock_read, solver
            )
            if if_solved != "No output":
                results[core], event[0] = if_solved

        if results[core][0] != int(0):
            sub_now = time.process_time()
            results[core][1] = results[core][1] + sub_now - sub_start[core]
            ev.set()
            event[0] = 1
            winner[0] = core
            res[core] = results[core][:]
            new_time[0] = results[core][1]
            awaiting = False

        if event[0] == 1 or ev.is_set():
            awaiting = False
            proc.terminate()
            time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
                time.sleep(0.1)
                for index in range(num_parameters):
                    if (
                        sub_start[index] - time.process_time() >= new_time[0]
                        and index != core
                    ):
                        os.kill(pids[index], signal.SIGKILL)
                time.sleep(0.1)
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    continue
            if solver == "cadical":
                time.sleep(0.1)
                for index in range(num_parameters):
                    if (
                        sub_start[index] - time.process_time() >= new_time[0]
                        and index != core
                    ):
                        try:
                            os.system("killall cadical")
                        except:
                            continue


def tournament(
    args,
    n,
    contender_list,
    start_run,
    filename,
    Pool,
    sub_start,
    pids,
    results,
    interim,
    ev,
    event,
    new_time,
):
    for core in range(n):
        contender = str(contender_list[core])

        param_string = set_param.set_contender_params(
            contender, Pool[contender], solver_parameters, return_it=True
        )

        # noinspection PyTypeChecker
        process[core] = mp.Process(
            target=start_run,
            args=[
                args,
                filename,
                args.timeout,
                param_string,
                core,
                sub_start,
                pids,
                results,
                interim,
                ev,
                event,
                new_time,
            ],
        )

    # Starting processes
    for core in range(n):
        process[core].start()

    return process


def watch_run(process, start, n, ev, pids):
    x = 0
    while any(proc.is_alive() for proc in process):
        time.sleep(1)
        x = x + 1
        if x == 4:
            x = 0
        now = time.time()
        currenttime = now - start
        if currenttime >= args.timeout:
            ev.set()
            event[0] == 1  # Question: What is it's use?
            for core in range(n):
                try:
                    os.kill(pids[core], signal.SIGKILL)
                except:
                    continue
        if ev.is_set() or event[0] == 1:
            if solver == "cadical":
                time.sleep(10)
                if any(proc.is_alive() for proc in process):
                    try:
                        os.system("killall cadical")
                    except:
                        continue


def close_run(n, interim, process, res, interim_res):
    # Prepare interim for processing (str -> int)
    for core in range(n):
        interim[core] = list(map(int, interim[core]))

    # Close processes
    for core in range(n):
        process[core].join()

    for core in range(n):
        results[core][:] = res[core][:]
        interim_res[core][:] = interim[core][:]

    return results, interim_res


def cppl_update(
    contender_pool,
    contender_list,
    winner,
    Y_t,  # Question: No use because it is not getting used in the function
    theta_hat,
    theta_bar,
    S_t,
    X_t,
    gamma_1,
    time_step,
    alpha,
):
    current_pool = []

    for keys in contender_pool:
        current_pool.append(contender_pool[keys])

    current_contender_names = []
    for i in range(len(contender_list)):
        current_contender_names.append(str(contender_pool[contender_list[i]]))

    contender_list = []
    for i in range(num_parameters):
        contender_list.append("contender_" + str(S_t[i]))

    Y_t = int(contender_list[winner[0]][10:])  # Y_t is getting written here, No use in passing as parameter
    print(f"Winner is contender_{Y_t}")
    [theta_hat, theta_bar, grad] = CPPLConfig.update(
        Y_t, theta_hat, theta_bar, S_t, X_t, gamma_1, time_step, alpha
    )

    return current_pool, current_contender_names, theta_hat, theta_bar, grad, Y_t


if __name__ == "__main__":

    (
        args,
        solver,
        directory,
        times_instances,
        problem_instance_list,
        tracking_times,
        tracking_pool,
        num_parameters,
        solver_parameters,
        contender_pool,
        training_files,
        train_list,
        rounds_to_train,
        standard_scaler,
        pca_obj_inst,
        params,
        parameter_value_dict,
        min_max_scaler,
        pca_obj_params,
        jfm,
        theta_hat,
        theta_bar,
        grad_op_sum,
        hess_sum,
        omega,
        gamma_1,
        alpha,
        time_step,
        winner_index_time_step,
        S_t,
        grad,
        winner_known,
    ) = _main()

    # Todo: Create a class CPPL which contains def run, cppl_update.
    run = True

    while run:
        # Iterate through all Instances
        for filename in problem_instance_list:

            # Read Instance file name to hand to solver
            # and check for format
            if solver == "cadical" or solver == "glucose":
                file_ending = ".cnf"
            elif solver == "cplex":
                file_ending = ".mps"  # Todo: add .mps file
            else:
                file_ending = None  # No solver is provided by default
            dot = filename.rfind(".")

            file_path = f"{directory}/" + str(filename)

            # Run parametrization on instances
            print(f"{filename[dot:]}, {file_ending}")
            if filename[dot:] == file_ending:  # and file_path not in train_list: # Only for training (Not further pursued

                print(
                    "\n \n ######################## \n",
                    "STARTING A NEW INSTANCE!",
                    "\n ######################## \n \n",
                )

                if winner_known:
                    # Get contender list from CPPLConfig.py
                    X_t, contender_list, discard = CPPLConfig.get_contenders(
                        directory=directory,
                        filename=filename,
                        pca_obj_inst=pca_obj_inst,
                        pca_obj_params=pca_obj_params,
                        jfm=jfm,
                        theta_bar=theta_bar,
                        time_step=time_step,
                        subset_size=num_parameters,
                        S_t=S_t,
                        grad=grad,
                        hess_sum=hess_sum,
                        grad_op_sum=grad_op_sum,
                        omega=omega,
                        solver=solver,
                        Pool=contender_pool,
                        tracking_Pool=tracking_pool,
                        min_max_scaler=min_max_scaler,
                        standard_scaler=standard_scaler,
                        parameter_limit=float(args.paramlimit),
                        param_value_dict=parameter_value_dict,
                        solver_parameters=solver_parameters,
                        exp=args.exp,
                    )

                    S_t = []
                    for i in range(len(contender_list)):
                        S_t.append(int(contender_list[i].replace("contender_", "")))

                    if discard:
                        time_step = 1

                    time_step = time_step + 1

                else:
                    contender_list = CPPLConfig.contender_list_including_generated(
                        contender_pool,
                        solver,
                        float(args.paramlimit),
                        params,
                        solver_parameters,
                        theta_bar,
                        jfm,
                        min_max_scaler,
                        pca_obj_params,
                        standard_scaler,
                        pca_obj_inst,
                        directory,
                        filename,
                        num_parameters,
                    )

                # Start run
                (
                    multiprocess_event,
                    event,
                    winner,
                    res,
                    interim,
                    new_time,
                    process_ids,
                    sub_start,
                    process,
                    results,
                    interim_res,
                    start,
                    winner_known,
                ) = _init_data_structures()

                process = tournament(
                    args=args,
                    n=num_parameters,
                    contender_list=contender_list,
                    start_run=start_run,
                    filename=file_path,
                    Pool=contender_pool,
                    sub_start=sub_start,
                    pids=process_ids,
                    results=results,
                    interim=interim,
                    ev=multiprocess_event,
                    event=event,
                    new_time=new_time,
                )

                # Output Setting
                if args.data == "y":
                    print("Prior contender data is used!\n")
                print("Timeout set to", args.timeout, "seconds\n")
                print("contender_pool size set to", args.contenders, "individuals\n")
                if args.pws == "pws":
                    print("Custom individual injected\n")
                else:
                    print("No custom Individual injected\n")
                print(".\n.\n.\n.\n")

                # Observe the run and stop it if one parameterization finished
                watch_run(process, start, num_parameters, multiprocess_event, process_ids)

                results, interim_res = close_run(
                    num_parameters, interim, process, res, interim_res
                )

                print(f"Instance {filename} was finished!\n")

                # Update CPPLConfig.py
                if args.baselineperf:
                    winner[0] = None
                    winner_known = False
                if winner[0] is not None:
                    (
                        current_pool,
                        current_contender_names,
                        theta_hat,
                        theta_bar,
                        grad,
                        winner_index_time_step,
                    ) = cppl_update(
                        contender_pool=contender_pool,
                        contender_list=contender_list,
                        winner=winner,
                        Y_t=winner_index_time_step,
                        theta_hat=theta_hat,
                        theta_bar=theta_bar,
                        S_t=S_t,
                        X_t=X_t,
                        gamma_1=gamma_1,
                        time_step=time_step,
                        alpha=alpha,
                    )
                else:
                    winner_known = False

                print("Time needed:", round(new_time[0], 2), "seconds\n\n\n")

                # Update solving times for instances
                times_instances.append(round(new_time[0], 2))

                # Log times needed for instances to file
                tracking_times.info(times_instances)

            # # Manage Training of args.train_number instances for args.train_rounds times (Not further Pursued)
            # if args.train_number is not None:
            #     files = sorted(training_files)
            #     if filename == files[int(args.train_number) - 1]:
            #         if rounds_to_train != 0:
            #             print(
            #                 "Training Round",
            #                 int(args.train_rounds - rounds_to_train + 1),
            #                 "Completed.\nRestart Run",
            #             )
            #             rounds_to_train = rounds_to_train - 1
            #             inst = inst - int(args.train_number)
            #             break
            #         else:
            #             run = False
            # else:
            #     run = False

        else:
            # When directory has no more instances, break
            break

print(
    "\n  #######################\n ",
    "Finished all instances!\n ",
    "#######################\n",
)
