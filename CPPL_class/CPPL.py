"""A CPPL class which represents CPPL algorithm"""
import csv
import json
import os
import pathlib
import random
import sys
import logging
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

import utils.utility_functions
from utils import random_genes, set_param, file_logging
from utils.constants import Constants
from preselection.upper_confidence_bound import UCB
from utils.utility_functions import (
    get_problem_instance_list,
    json_validation,
    set_genes,
    join_feature_map,
    gradient,
)
from utils.log_params_utils import log_space_convert
from tournament_classes.tournament import Tournament


class CPPLBase:
    """A generic CPPL Class which includes initialization, running and updating the parameters in CPPL algorithm.

    Parameters
    ----------
    args: Namespace
        The namespace variable for all the arguments in the parser.
    """

    def __init__(
            self,
            args,
            logger_name="CPPLBase",
            logger_level=logging.INFO,
    ):
        self.args = args
        self.subset_size = (
            mp.cpu_count()
        )  # subset size (k) in algorithm =  number of cpu available
        self.parameter_limit = float(args.paramlimit)
        self.instance_execution_times = []

        # Creating tracking logs
        self.tracking_times = file_logging.tracking_files(
            filename=f"{self.args.times_file_name}_{self.args.solver}.txt",
            logger_name="CPPL_Times",
            level=logger_level,
        )
        self.tracking_winners = file_logging.tracking_files(
            filename=f"Winners_{self.args.solver}.txt",
            logger_name="CPPL_Winners",
            level=logger_level,
        )
        self.tracking_pool = file_logging.tracking_files(
            filename=f"Pool_{self.args.solver}.json",
            logger_name="CPPL_Pool",
            level=logger_level,
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        # Initialization
        self._init_checks()  # Check whether folders exist for algorithm run and check the args values
        self.solver_parameters = (
            self._get_solver_parameters()
        )  # get parameters from the solver's json file
        self._init_pool()  # Initialize contender pool

        self.tracking_pool.info(
            self.contender_pool
        )  # Write contender_pool to textfile for solver to access parameter settings

        # initialize features from instance files
        (
            self.standard_scalar,
            self.n_pca_features_components,
            self.pca_obj_instances,
            self.train_list,
        ) = self._init_pca_features()
        self.cppl_utils = CPPLUtils(
            pool=self.contender_pool,
            solver=self.args.solver,
            solver_parameters=self.solver_parameters,
        )
        (
            self.params,
            self.parameter_value_dict,
            self.min_max_scalar,
            self.n_pca_params_components,
            self.pca_obj_params,
        ) = self._init_pca_params()
        # other parameters
        candidates_pool_dimensions = self._get_candidates_pool_dimensions()

        # theta_hat = np.random.rand(candidates_pool_dimensions)
        self.theta_hat = np.zeros(
            candidates_pool_dimensions
        )  # Line 2 CPPL (random Parameter Vector)
        self.theta_bar = self.theta_hat  # Line 3 CPPl
        self.grad_op_sum = np.zeros(
            (candidates_pool_dimensions, candidates_pool_dimensions)
        )
        self.hess_sum = np.zeros(
            (candidates_pool_dimensions, candidates_pool_dimensions)
        )
        self.grad = np.zeros(
            candidates_pool_dimensions
        )  # Gradient ∇L in line 11 to update ˆθt

        self.omega = (
            self.args.omega
        )  # Parameter of CPPL *Helps determine the confidence intervals (Best value = 0.001)
        self.gamma = self.args.gamma  # Parameter CPPL (Best value = 1)
        self.alpha = self.args.alpha  # Parameter CPPL (Best value = 0.2)
        self.time_step = 0  # initial time step = 0 where initialization takes place
        self.Y_t = 0  # The Winning contender, the top-ranked arm among the subset St provided by the underlying
        # contextualized PL model in case of winner feedback
        self.S_t = []  # Subset of contenders Line 9 of CPPL

        self.winner_known = True
        self.is_finished = False

    def _init_checks(self):

        # Check Instance problem directory in arguments
        if self.args.directory != "No Problem Instance Directory given":
            self._init_output_files()
        else:
            print(
                "\n\nYou need to specify a directory containing the problem instances!\n\n**[-d directory_name]**\n\n"
            )
            sys.exit(0)

        # Check if the Instance feature directory available
        instance_feature_directory = pathlib.Path("Instance_Features")
        if instance_feature_directory.exists():
            pass
        else:
            print(
                "\nWARNING!\n\nA directory named <Instance_Features> with a "
                ".csv file containing the instance features is necessary!"
            )

            if self.args.directory != "No Problem Instance Directory given":
                print(
                    "\nIt must be named: Features_" + str(self.directory)[2:-1] + ".csv"
                )

            else:
                print(
                    "\nIt must be named: "
                    "Features_<problem_instance_directory_name>.csv"
                )
            print(
                "\ninstance1 feature_value1 feature_value2 "
                "....\ninstance2 feature_value1..."
            )
            print("\nExiting...")
            sys.exit(0)

        # Check Solver in arguments
        if self.args.solver == "No solver chosen":
            print("\nYou need to choose a solver!!!\n\n**[-s <solver_name>]**\n\n")
            sys.exit(0)
        else:
            parameter_directory = pathlib.Path(f"{Constants.PARAM_POOL_FOLDER.value}")
            if parameter_directory.exists():
                pass
            else:
                os.mkdir(f"{Constants.PARAM_POOL_FOLDER.value}")

        # Check training is required
        if self.args.train_number is not None:
            self.training = True
            for self.root, self.training_directory, self.training_file in sorted(
                    os.walk(self.directory)
            ):
                continue
        else:
            self.training = False

        if self.args.train_rounds is not None:
            self.rounds_to_train = int(self.args.train_rounds)
        else:
            self.rounds_to_train = 0

    def _get_solver_parameters(self) -> dict:
        json_file_name = "params_" + str(self.args.solver)

        with open(f"{Constants.PARAMS_JSON_FOLDER}/{json_file_name}.json", "r") as file:
            data = file.read()
        parameters = json.loads(data)
        param_names = list(parameters.keys())

        with open(
                f"{Constants.PARAMS_JSON_FOLDER.value}/{Constants.PARAM_SCHEMA_JSON_FILE.value}",
                "r",
        ) as file:
            schema = file.read()
        schema_meta = json.loads(schema)

        for parameter_name in param_names:
            valid = json_validation(
                param=parameters[parameter_name], schema=schema_meta
            )
            if not valid:
                print("Invalid JSON data structure. Exiting.\n\n")
                print(f"{parameters[parameter_name]}")
                sys.exit(0)
        return parameters

    def _init_pool(self):
        if self.args.data is None:
            pool_keys = [f"contender_{c}" for c in range(self.args.contenders)]
            self.contender_pool = dict.fromkeys(pool_keys, 0)

            if self.args.baselineperf:
                print("Baseline Performance Run (onl%s default parameters)")
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = set_genes(
                        solver_parameters=self.solver_parameters
                    )
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters,
                    )
            else:
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = random_genes.get_genes_set(
                        solver=self.args.solver
                    )
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters,
                    )

                if self.args.pws is not None:
                    contender_index = "contender_0"
                    self.contender_pool[contender_index] = set_genes(
                        solver_parameters=self.solver_parameters
                    )
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters,
                    )  # Create the contenders pool in a directory
        elif self.args.data == "y":
            pool_file = f"Pool_{self.args.solver}.json"
            if self.args.exp == "y":
                pool_file = f"Pool_exp_{self.args.solver}.json"
            with open(f"{pool_file}", "r") as file:
                self.contender_pool = eval(file.read())
                for contender_index in self.contender_pool:
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters,
                    )

    def _init_output_files(self):
        self.directory = os.fsencode(self.args.directory)
        # path, dirs, files = next(os.walk(self.args.directory))
        if self.args.file_order == "ascending":
            self.problem_instance_list = get_problem_instance_list(
                sorted_directory=sorted(os.listdir(self.directory))
            )
        elif self.args.file_order == "descending":
            self.problem_instance_list = get_problem_instance_list(
                sorted(os.listdir(self.directory), reverse=True)
            )
        else:
            file_order = str(self.args.file_order)
            with open(f"{file_order}.txt", "r") as file:
                self.problem_instance_list = eval(file.read())
        with open(f"{Constants.PROBLEM_INSTANCE_LIST_TXT_FILE.value}", "w") as file:
            print(
                self.problem_instance_list, file=file
            )  # Print all the instance in problem_instance_list.txt

    def _init_pca_features(self):
        # read features
        if os.path.isfile(
                "Instance_Features/training_features_" + str(self.directory)[2:-1] + ".csv"
        ):
            pass
        else:
            print(
                "\n\nThere needs to be a file with training instance features "
                "named << training_features_" + str(self.directory)[2:-1] + ".csv >> in"
                                                                            " the directory Instance_Features\n\n"
            )
            sys.exit(0)

        self.directory = str(self.directory)[2:-1]

        # Read features from .csv file
        features, train_list = self._read_training_features_csv()

        standard_scalar = StandardScaler()
        features = standard_scalar.fit_transform(features)

        # PCA on features
        n_pca_features_components = self.args.nc_pca_f
        pca_obj_instances = PCA(n_components=n_pca_features_components).fit(features)

        return standard_scalar, n_pca_features_components, pca_obj_instances, train_list

    def _read_training_features_csv(self):
        features = []
        train_list = []
        with open(
                f"Instance_Features/training_features_{self.directory}.csv", "r"
        ) as csvFile:
            reader = csv.reader(csvFile)
            next(reader)
            for row in reader:
                if len(row[0]) != 0:
                    next_features = row
                    train_list.append(
                        row[0]
                    )  # TODO remove after clarification about training
                    next_features.pop(0)
                    next_features = [float(j) for j in next_features]
                    features.append(next_features)
        csvFile.close()

        return np.asarray(features), train_list

    def _init_pca_params(self):

        params, parameter_value_dict = self.cppl_utils.read_parameters()
        params = np.asarray(params)
        all_min, all_max = random_genes.get_all_min_and_max(self.solver_parameters)
        all_min, _ = self.cppl_utils.read_parameters(contender=all_min)
        all_max, _ = self.cppl_utils.read_parameters(contender=all_max)
        params = np.append(params, [all_min, all_max], axis=0)
        params = log_space_convert(
            limit_number=float(self.args.paramimit),
            param_set=params,
            solver_parameter=self.solver_parameters,
        )
        min_max_scalar = MinMaxScaler()
        params = min_max_scalar.fit_transform(params)
        n_pca_params_components = self.args.nc_pca_p
        pca_obj_params = PCA(n_components=n_pca_params_components)
        pca_obj_params.fit(params)

        return (
            params,
            parameter_value_dict,
            min_max_scalar,
            n_pca_params_components,
            pca_obj_params,
        )

    def _get_candidates_pool_dimensions(self):
        self.joint_featured_map_mode = self.args.jfm  # Default="polynomial"
        candidates_pool_dimensions = 4  # by default # Corresponds to n different parameterization (Pool of candidates)
        if self.joint_featured_map_mode == "concatenation":
            candidates_pool_dimensions = (
                    self.n_pca_features_components + self.n_pca_params_components
            )
        elif self.joint_featured_map_mode == "kronecker":
            candidates_pool_dimensions = (
                    self.n_pca_features_components * self.n_pca_params_components
            )
        elif self.joint_featured_map_mode == "polynomial":
            for index_pca_params in range(
                    (self.n_pca_features_components + self.n_pca_params_components) - 2
            ):
                candidates_pool_dimensions = (
                        candidates_pool_dimensions + 3 + index_pca_params
                )

        return candidates_pool_dimensions

    def get_context_feature_matrix(self, filename):
        # read and preprocess instance features (PCA)
        features = self.get_features(filename=f"{filename}")
        features = self.standard_scalar.transform(features.reshape(1, -1))
        features = self.pca_obj_instances.transform(features)
        # get parametrization
        params, _ = self.cppl_utils.read_parameters()
        params = np.asarray(params)
        params = log_space_convert(
            limit_number=self.parameter_limit,
            param_set=params,
            solver_parameter=self.solver_parameters,
        )
        params = self.min_max_scalar.transform(params)
        # PCA on parametrization
        params_transformed = self.pca_obj_params.transform(params)
        # construct X_t (context specific (instance information) feature matrix ( and parameterization information))
        n_arms = params.shape[0]  # Distinct Parameters or available arms
        degree_of_freedom = len(
            self.theta_bar
        )  # Distinct Parameters or available arms
        context_matrix = np.zeros((n_arms, degree_of_freedom))
        for i in range(n_arms):
            next_context_vector = join_feature_map(
                x=params_transformed[
                    i,
                ],
                y=features[0],
                mode=self.joint_featured_map_mode,
            )
            context_matrix[i, :] = next_context_vector
        # Normalizing the context specific features
        normalize(
            context_matrix, norm="max", copy=False
        )  # TODO add "X_t =" after clearing the doubt
        # compute estimated contextualized utility parameters (v_hat)
        v_hat = np.zeros(n_arms)  # Line 7 in CPPL algorithm
        for i in range(n_arms):
            v_hat[i] = np.exp(np.inner(self.theta_bar, context_matrix[i, :]))
        return context_matrix, degree_of_freedom, features, n_arms, params, v_hat

    def get_features(self, filename):
        with open(f"Instance_Features/Features_{self.directory}.csv", "r") as csvFile:
            reader = csv.reader(csvFile)
            next(reader)
            for row in reader:
                row_list = row[1:]
                features = [float(j) for j in row_list]
                if os.path.basename(row_list[0]) == filename:
                    csvFile.close()
                    break
        return np.asarray(features)


class CPPLConfiguration:
    def __int__(
            self,
            args,
            logger_name="CPPLConfiguration",
            logger_level=logging.INFO,
    ):
        self.base = CPPLBase(args=args)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.filename = None
        self.async_results = []

    def _get_contender_list(self, filename: str):
        """

        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        self.filename = filename
        (
            X_t,  # Context matrix
            degree_of_freedom,  # context vector dimension (len of theta_bar)
            self.pca_context_features,
            self.n_arms,  # Number of parameters
            self.params,
            v_hat,
        ) = self.base.get_context_feature_matrix(filename=self.filename)
        self.discard = []

        ucb = UCB(
            cppl_base_object=self.base,
            context_matrix=X_t,
            degree_of_freedom=degree_of_freedom,
            n_arms=self.n_arms,
            v_hat=v_hat,
        )
        if self.base.time_step == 0:
            (
                self.base.S_t,
                self.contender_list_str,
                v_hat,
            ) = ucb.run()  # Get the subset S_t

            if self.base.args.exp == "y":
                for i in range(self.base.subset_size):
                    self.contender_list_str[i] = f"contender_{i}"

            genes = set_genes(solver_parameters=self.base.solver_parameters)
            set_param.set_contender_params(
                contender_index="contender_0",
                contender_pool=genes,
                solver_parameters=self.base.solver_parameters,
            )
            self.contender_list_str[0] = "contender_0"
            self.update_logs(contender_index="0", genes=genes)

        else:
            # compute confidence_t and select S_t (symmetric group on [num_parameters], consisting of rankings: r ∈ S_n)
            self.base.S_t, confidence_t, self.contender_list_str, v_hat = ucb.run()

            # update contenders
            for arm1 in range(self.n_arms):
                for arm2 in range(self.n_arms):
                    if (
                            arm2 != arm1
                            and v_hat[arm2] - confidence_t[arm2]
                            >= v_hat[arm1] + confidence_t[arm1]
                            and (not arm1 in self.discard)
                    ):
                        self.discard.append(arm1)
                        break
            if len(self.discard) > 0:
                self.contender_list_str = self._generate_new_parameters()

        return X_t, self.contender_list_str, self.discard

    def _generate_new_parameters(self):
        """

        Returns
        -------

        """
        print(f"There are {len(self.discard)} parameterizations to discard")
        discard_size = len(self.discard)

        print(
            "\n *********************************",
            "\n Generating new Parameterizations!",
            "\n *********************************\n",
        )
        self.new_candidates_size = 1000  # generate with randomenss
        random_parameters, _ = self.base.cppl_utils.read_parameters()
        random_parameters = np.asarray(random_parameters)
        random_parameters = log_space_convert(
            limit_number=self.base.parameter_limit,
            param_set=random_parameters,
            solver_parameter=self.base.solver_parameters,
        )

        self.best_candidate = log_space_convert(
            limit_number=self.base.parameter_limit,
            param_set=random_genes.get_one_hot_decoded_param_set(
                genes=random_parameters[self.base.S_t[0]],
                solver=self.base.args.solver,
                param_value_dict=self.base.parameter_value_dict,
                solver_parameters=self.base.solver_parameters,
            ),
            solver_parameter=self.base.solver_parameters,
            exp=True,
        )

        self.second_candidate = log_space_convert(
            limit_number=self.base.parameter_limit,
            param_set=random_genes.get_one_hot_decoded_param_set(
                genes=random_parameters[self.base.S_t[1]],
                solver=self.base.args.solver,
                param_value_dict=self.base.parameter_value_dict,
                solver_parameters=self.base.solver_parameters,
            ),
            solver_parameter=self.base.solver_parameters,
            exp=True,
        )

        (
            new_candidates_transformed,
            new_candidates,
        ) = self._parallel_evolution_and_fitness()
        new_candidates_transformed = self.base.min_max_scalar.transform(
            new_candidates_transformed
        )
        new_candidates_transformed = self.base.pca_obj_params.transform(
            new_candidates_transformed
        )
        v_hat_new_candidates = np.zeros(self.new_candidates_size)

        for index in range(self.new_candidates_size):
            context_vector = join_feature_map(
                x=new_candidates_transformed[index],
                y=self.pca_context_features,
                mode=self.base.joint_featured_map_mode,
            )

            v_hat_new_candidates[index] = np.exp(
                np.inner(self.base.theta_bar, context_vector)
            )

        best_new_candidates_list = (-v_hat_new_candidates).argsort()[0:discard_size]

        for index, _ in enumerate(best_new_candidates_list):
            best_new_candidate_param_set = random_genes.get_one_hot_decoded_param_set(
                genes=new_candidates[best_new_candidates_list[index]],
                solver=self.base.args.solver,
                param_value_dict=self.base.parameter_value_dict,
                solver_parameters=self.base.solver_parameters,
            )
            genes = random_genes.get_params_string_from_numeric_params(
                genes=log_space_convert(
                    limit_number=self.base.parameter_limit,
                    param_set=best_new_candidate_param_set,
                    solver_parameter=self.base.solver_parameters,
                    exp=True,
                ),
                solver=self.base.args.solver,
                solver_parameters=self.base.solver_parameters,
            )

            if type(self.discard) != list:
                self.discard = np.asarray(self.discard)

            set_param.set_contender_params(
                contender_index="contender_" + str(self.discard[index]),
                contender_pool=genes,
                solver_parameters=self.base.solver_parameters,
            )
            self.update_logs(contender_index=self.discard[index], genes=genes)

        new_contender_list = self._contender_list_including_generated()

        return new_contender_list

    # TODO convert it into a class
    def _parallel_evolution_and_fitness(self):
        candidate_pameters = random_genes.get_one_hot_decoded_param_set(
            genes=self.params[self.base.S_t[0]],
            solver=self.base.args.solver,
            param_value_dict=self.base.parameter_value_dict,
            solver_parameters=self.base.solver_parameters,
        )
        self.candidate_pameters_size = len(candidate_pameters)
        self.new_candidates = np.zeros(
            shape=(self.new_candidates_size, self.candidate_pameters_size)
        )

        last_step = self.new_candidates_size % self.base.subset_size
        step_size = (self.new_candidates_size - last_step) / self.base.subset_size
        self.all_steps = []

        for _ in range(self.base.subset_size):
            self.all_steps.append(int(step_size))
        self.all_steps.append(int(last_step))

        step = 0
        pool = mp.Pool(processes=self.base.subset_size)

        for index, _ in enumerate(self.all_steps):
            step += self.all_steps[index]
            pool.apply_async(
                func=self._evolution_and_fitness,
                args=(self.all_steps[index]),
                callback=self.save_result,
            )
        pool.close()
        pool.join()

        new_candidates_transformed = []
        new_candidates = []

        for i, _ in enumerate(self.async_results):
            for j, _ in enumerate(self.async_results[i][0]):
                new_candidates_transformed.append(self.async_results[i][0][j])
                new_candidates.append(self.async_results[i][1][j])

        return new_candidates_transformed, new_candidates

    def _evolution_and_fitness(self, new_candidates_size):
        # Generation approach based on genetic mechanism with mutation and random individuals
        new_candidates = np.zeros(
            shape=(new_candidates_size, len(self.new_candidates[0]))
        )

        for candidate in range(self.new_candidates_size):
            random_individual = random.uniform(0, 1)
            next_candidate = np.zeros(self.candidate_pameters_size)
            contender = random_genes.get_genes_set(
                solver=self.base.args.solver,
                solver_parameters=self.base.solver_parameters,
            )
            genes, _ = self.base.cppl_utils.read_parameters(contender=contender)
            mutation_genes = random_genes.get_one_hot_decoded_param_set(
                genes=genes,
                solver=self.base.args.solver,
                param_value_dict=self.base.parameter_value_dict,
                solver_parameters=self.base.solver_parameters,
            )

            for index in range(self.candidate_pameters_size):
                random_seed = random.uniform(0, 1)
                mutation_seed = random.uniform(0, 1)

                # Dueling function
                if random_seed > 0.5:
                    next_candidate[index] = self.best_candidate[index]
                else:
                    next_candidate[index] = self.second_candidate[candidate]
                if mutation_seed < 0.1:
                    next_candidate[index] = mutation_genes[index]

            if random_individual < 0.99:
                new_candidates[candidate] = mutation_genes
            else:
                new_candidates[candidate] = next_candidate

        new_candidates = random_genes.get_one_hot_decoded_param_set(
            genes=new_candidates,
            solver=self.base.args.solver,
            solver_parameters=self.base.solver_parameters,
            reverse=True,
        )
        new_candidates_transformed = log_space_convert(
            limit_number=self.base.parameter_limit,
            param_set=new_candidates,
            solver_parameter=self.base.solver_parameters,
        )

        return new_candidates_transformed, new_candidates

    def save_result(self, result):
        new_candidates_transformed = result[0]
        new_candidates = result[1]
        self.async_results.append([new_candidates_transformed, new_candidates])

    ##################################################################################

    def _contender_list_including_generated(self):
        contender_list = []

        _, _, _, _, _, v_hat = self.base.get_context_feature_matrix(
            filename=self.filename
        )

        S_t = (-v_hat).argsort()[0: self.base.subset_size]

        for index in range(self.base.subset_size):
            contender_list.append("contender_" + str(S_t[index]))

        return contender_list

    def update_logs(self, contender_index, genes):
        self.base.contender_pool[
            "contender_" + str(contender_index)
            ] = genes  # TODO: change to local pool after clarification
        self.base.tracking_pool.info(self.base.contender_pool)


class CPPLAlgo(CPPLConfiguration):
    def __init__(
            self,
            args,
            logger_name="CPPLAlgo",
            logger_level=logging.INFO,
    ):
        super().__init__(args=args)
        self.tournament = None
        self.contender_list = None
        self.context_matrix = None
        self.current_contender_names = None
        self.current_pool = None
        self.solver = self.base.args.solver
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.winners_list = []

    def run(self):
        # Read Instance file name to hand to solver
        # and check for format
        if self.solver == "cadical" or self.solver == "glucose":
            file_ending = ".cnf"
        else:
            file_ending = ".mps"

        while not self.base.is_finished:
            # Iterate through all Instances
            for filename in self.base.problem_instance_list:

                dot = filename.rfind(".")
                file_path = f"{self.base.directory}/" + str(filename)

                # Run parametrization on instances
                if (
                        filename[dot:] == file_ending
                ):  # Check if input file extension is same as required by solver
                    print(
                        "\n \n ######################## \n",
                        "STARTING A NEW INSTANCE!",
                        "\n ######################## \n \n",
                    )

                    if self.base.winner_known:
                        # Get contender list
                        # X_t: Context information
                        # Y_t: winner
                        # S_t: subset of contenders
                        (
                            self.context_matrix,
                            self.contender_list,
                            discard,
                        ) = self._get_contender_list(filename=filename)

                        self.base.S_t = []  # S_t
                        for contender in self.contender_list:
                            self.base.S_t.append(
                                int(contender.replace("contender_", ""))
                            )

                        if discard:
                            self.base.time_step = 1
                        self.base.time_step += 1
                    else:
                        self.contender_list = self._contender_list_including_generated()

                    self.tournament = Tournament(
                        cppl_base=self.base,
                        filepath=file_path,
                        contender_list=self.contender_list,
                    )
                    self.tournament.run()

                    # Output Setting
                    if self.base.args.data == "y":
                        print("Prior contender data is used!\n")
                    print("Timeout set to", self.base.args.timeout, "seconds\n")
                    print(
                        "contender_pool size set to",
                        self.base.args.contenders,
                        "individuals\n",
                    )
                    if self.base.args.pws == "pws":
                        print("Custom individual injected\n")
                    else:
                        print("No custom Individual injected\n")
                    print(".\n.\n.\n.\n")

                    # Observe the run and stop it if one parameterization finished
                    self.tournament.watch_run()
                    self.tournament.close_run()

                    print(f"Instance {filename} was finished!\n")

                    # Update parameter set
                    if self.base.args.baselineperf:
                        self.tournament.winner[0] = None
                        self.base.winner_known = False

                    if self.tournament.winner[0] is not None:
                        self.update()
                    else:
                        self.base.winner_known = False

                    print(
                        f"Time needed: {round(self.tournament.new_time[0], 2)} seconds \n\n"
                    )

                    # Update solving times for instances
                    self.base.instance_execution_times.append(
                        round(self.tournament.new_time[0], 2)
                    )

                    # Log execution times
                    self.base.tracking_times.info(self.base.instance_execution_times)

                    # Log Winners for instances
                    self.base.tracking_winners.info(self.winners_list)

                else:
                    # When directory has no more instances, break
                    self.base.is_finished = True

        print(
            "\n  #######################\n ",
            "Finished all instances!\n ",
            "#######################\n",
        )

    def update(self):
        self.current_pool = []

        for keys in self.base.contender_pool:
            self.current_pool.append(self.base.contender_pool[keys])

        self.current_contender_names = []
        for index, _ in enumerate(self.contender_list):
            self.current_contender_names.append(
                str(self.base.contender_pool[self.contender_list[index]])
            )

        self.contender_list = []
        for i in range(self.base.subset_size):
            self.contender_list.append(f"contender_{str(self.base.S_t[i])}")
        self.base.Y_t = int(self.contender_list[self.tournament.winner[0]][10:])
        print(f"Winner is contender_{self.base.Y_t}")
        self.winners_list.append(self.base.Y_t)  # Track winners

        self.base.grad = gradient(
            theta=self.base.theta_hat,
            winner_arm=self.base.Y_t,
            subset_arms=self.base.S_t,
            context_matrix=self.context_matrix,
        )

        self.base.theta_hat = (
                self.base.theta_hat
                + self.base.gamma
                * self.base.time_step ** (-self.base.alpha)
                * self.base.grad
        )
        self.base.theta_hat[self.base.theta_hat < 0] = 0
        self.base.theta_hat[self.base.theta_hat > 0] = 1

        # Update theta_bar
        self.base.theta_bar = (
                (self.base.time_step - 1) * self.base.theta_bar / self.base.time_step
                + self.base.theta_hat / self.base.time_step
        )


class CPPLUtils:
    def __init__(
            self,
            pool,
            solver,
            solver_parameters,
            logger_name="CPPLUtils",
            logger_level=logging.INFO,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.pool = pool
        self.solver = solver
        self.solver_parameters = solver_parameters

    def read_parameters(self, contender=None):
        global parameter_value_dict
        parameter_names, params = utils.utility_functions.get_solver_params(
            solver_parameters=self.solver_parameters, solver=self.solver
        )

        if contender is not None:
            new_params, parameter_value_dict = self.read_param_from_dict(
                contender=contender, parameter_names=parameter_names
            )
            return np.asarray(new_params), parameter_value_dict

        else:
            if type(self.pool) != dict:
                contender_pool_list = list(self.pool)
                new_contender_pool = {}
                for i, _ in enumerate(contender_pool_list):
                    new_contender_pool["contender_" + str(i)] = contender_pool_list[i]
            else:
                new_contender_pool = self.pool

            new_params_list = []
            for new_contender in new_contender_pool:
                new_params, parameter_value_dict = self.read_param_from_dict(
                    contender=new_params_list[new_contender],
                    parameter_names=parameter_names,
                )
                new_params_list.append(new_params)
            return np.asarray(new_params_list), parameter_value_dict

    def read_param_from_dict(self, contender, parameter_names):
        global index
        next_params = contender
        params = self.solver_parameters
        originals_index_to_delete = []
        one_hot_addition = []
        parameter_value_dict = {}

        if len(parameter_names) == len(contender):
            for i, _ in enumerate(parameter_names):
                parameter_value_dict[parameter_names[i]] = next_params[i]

                if params[parameter_names[i]]["paramtype"] == "categorical":
                    if params[parameter_names[i]]["valtype"] == "int":
                        min_value = params[parameter_names[i]]["minval"]
                        max_value = params[parameter_names[i]]["maxval"]
                        value_range = max_value - min_value + 1
                        values = [min_value + j for j in range(value_range)]
                        index = int(next_params[i]) - min_value

                    else:  # params[parameter_names[i]]["valtype"] == "str"
                        values = params[parameter_names[i]]["values"]

                        if type(next_params[i]) == str:
                            index = values.index(next_params[i])
                        else:
                            pass
                    if len(values) == 2:
                        # The categorical Parameter can be treated as binary
                        # Replace original value by zero or one
                        for j in range(len(values)):
                            if next_params[i] == values[j]:
                                parameter_value_dict[parameter_names[i]] = j

                    elif len(values) > 2:
                        # The categorical Parameter needs One-Hot Encoding
                        # -> append One-Hot Vectors and delete original elements
                        one_hot = [0 for _, _ in enumerate(values)]

                        one_hot[index] = 1

                        originals_index_to_delete.append(i)
                        one_hot_addition += one_hot

                        parameter_value_dict[parameter_names[i]] = None

            new_params = []

            for key in parameter_value_dict:
                if parameter_value_dict[key] is not None:
                    new_params.append(parameter_value_dict[key])

            new_params += one_hot_addition

        else:
            new_params = contender

        return new_params, parameter_value_dict
