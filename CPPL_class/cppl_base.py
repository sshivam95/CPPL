"""The Base class of CPPL algorithm."""
from argparse import Namespace
import csv
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

from CPPL_class.cppl_utils import CPPLUtils
from utils import file_logging, set_param, random_genes
from utils.constants import Constants
from utils.log_params_utils import log_space_convert
from utils.utility_functions import (
    json_validation,
    set_genes,
    get_problem_instance_list,
    join_feature_map,
)


class CPPLBase:
    """A generic CPPL Class which includes initialization, running and updating the parameters in CPPL algorithm.

    Parameters
    ----------
    args: Namespace
        The namespace variable for all the arguments in the parser.
    logger_name : str, optional
            _description_, by default "CPPLBase"
    logger_level : int, optional
        _description_, by default logging.INFO

    Arguments
    ---------

    """

    def __init__(
        self,
        args: Namespace,
        logger_name: str = "CPPLBase",
        logger_level: int = logging.INFO,
    ):
        self.args = args
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

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
            candidates_pool_dimensions  # Estimated score/weight parameter
        )  # Line 2 CPPL (random Parameter Vector)
        self.theta_bar = self.theta_hat  # Line 3 CPPl, mean weight parameter
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

    def _init_checks(self) -> None:
        """_summary_"""
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

    def _get_solver_parameters(self) -> Dict:
        """_summary_

        Returns
        -------
        Dict
            _description_
        """
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

    def _init_pool(self) -> None:
        """_summary_"""
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

    def _init_output_files(self) -> None:
        """_summary_"""
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

    def _init_pca_features(self) -> Tuple[StandardScaler, int, PCA, List]:
        """_summary_

        Returns
        -------
        Tuple[StandardScaler, int, PCA, List]
            _description_
        """
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

    def _read_training_features_csv(self) -> Tuple[np.ndarray, List]:
        """_summary_

        Returns
        -------
        Tuple[np.ndarray, List]
            _description_
        """
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

    def _init_pca_params(self) -> Tuple[np.ndarray, Dict, MinMaxScaler, int, PCA]:
        """_summary_

        Returns
        -------
        Tuple[np.ndarray, Dict, MinMaxScaler, int, PCA]
            _description_
        """
        params, parameter_value_dict = self.cppl_utils.read_parameters()
        params = np.asarray(params)
        all_min, all_max = random_genes.get_all_min_and_max(self.solver_parameters)
        all_min, _ = self.cppl_utils.read_parameters(contender_genes=all_min)
        all_max, _ = self.cppl_utils.read_parameters(contender_genes=all_max)
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

    def _get_candidates_pool_dimensions(self) -> int:
        """_summary_

        Returns
        -------
        int
            _description_
        """
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

    def get_context_feature_matrix(
        self, filename: str
    ) -> Tuple[np.ndarray, int, np.ndarray, int, np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        filename : str
            _description_

        Returns
        -------
        Tuple[np.ndarray, int, np.ndarray, int, np.ndarray, np.ndarray]
            _description_
        """
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
        degree_of_freedom = len(self.theta_bar)  # Distinct Parameters or available arms
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

    def get_features(self, filename: str) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        filename : str
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
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
