"""The Base class of CPPL Framework."""
import csv
import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from CPPL_class.cppl_utils import CPPLUtils
from utils import file_logging, random_genes
from utils.constants import Constants
from utils.log_params_utils import log_space_convert
from utils.utility_functions import (
    get_problem_instance_list,
    join_feature_map,
    json_validation,
    set_genes,
)


class CPPLBase:
    """A generic CPPL Class which includes initialization, running and updating the parameters in CPPL framework.

    Parameters
    ----------
    args: Namespace
        The namespace variable for all the arguments in the parser.
    """

    def __init__(
        self,
        args: Namespace,
        logger_level: int = logging.INFO,
    ):
        self.args = args

        self.pool_filename = (
            f"Pool_{self.args.solver}.json"  # To track the solver's parameterizations
        )
        if self.args.exp == "y":
            self.pool_filename = f"Pool_exp_{self.args.solver}.json"

        # number of arms (n)
        self.num_contenders = self.args.contenders

        # subset size (k) in algorithm =  number of CPU cores available
        self.subset_size = mp.cpu_count()

        # Check if there are more CPU cores than the number of arms
        if self.subset_size > self.num_contenders:
            self.subset_size = int(self.num_contenders / 2)

        # Limit for the possible absolute value of a parameter for it to be normed to log space before CPPL computation
        self.parameter_limit = float(args.paramlimit)
        self.instance_execution_times = []

        # Creating tracking logs
        # Track the execution timings of each problem instance
        self.tracking_times = file_logging.tracking_files(
            filename=f"{self.args.times_file_name}_{self.args.solver}.txt",
            logger_name="CPPL_Times",
            level=logger_level,
        )
        # Track which contender won for each problem instance
        self.tracking_winners = file_logging.tracking_files(
            filename=f"Winners_{self.args.solver}.txt",
            logger_name="CPPL_Winners",
            level=logger_level,
        )
        # Track the parameterization of the solver which won in each problem instance
        self.tracking_pool = file_logging.tracking_files(
            filename=self.pool_filename,
            logger_name="CPPL_Pool",
            level=logger_level,
        )

        # Initialization
        self._init_checks()

        # Get the solver's parameter set (λ)
        self.solver_parameters = self._get_solver_parameters()

        # Initialize contender pool
        self._init_pool()

        # Write contender_pool to textfile for solver to access parameter settings
        self.tracking_pool.info(self.contender_pool)

        # Get the problem instance features
        (
            self.standard_scalar,
            self.n_pca_features_components,
            self.pca_obj_instances,
        ) = self._init_pca_features()

        # Initialize the utilitity object used in the CPPL framework
        self.cppl_utils = CPPLUtils(
            pool=self.contender_pool,
            solver=self.args.solver,
            solver_parameters=self.solver_parameters,
        )

        # Apply PCA on the solver's parameters
        (
            self.params,
            self.parameter_value_dict,
            self.min_max_scalar,
            self.n_pca_params_components,
            self.pca_obj_params,
        ) = self._init_pca_params()

        # Get the context vector
        candidates_pool_dimensions = self._get_candidates_pool_dimensions()

        # Estimated score/weight parameter
        # Line 2 of CPPL algorithm
        self.theta_hat = np.random.rand(candidates_pool_dimensions)
        # self.theta_hat = np.zeros(
        #     candidates_pool_dimensions
        # )
        # Line 3 CPPl, mean weight parameter
        # Initially both the quantities should be same
        self.theta_bar = self.theta_hat
        self.grad_op_sum = np.zeros(
            (candidates_pool_dimensions, candidates_pool_dimensions)
        )
        self.hess_sum = np.zeros(
            (candidates_pool_dimensions, candidates_pool_dimensions)
        )
        self.grad = np.zeros(candidates_pool_dimensions)

        # Parameter of CPPL *Helps determine the confidence intervals (Best value = 0.001)
        self.omega = self.args.omega

        # Parameter CPPL (Best value = 1)
        self.gamma = self.args.gamma

        # Parameter CPPL (Best value = 0.2)
        self.alpha = self.args.alpha

        # initial time step = 0 where initialization takes place
        self.time_step = 0

        # The Winning contender, the top-ranked arm among the subset S_t provided by the
        # underlying contextualized PL model in case of winner feedback
        self.Y_t = 0

        # Subset of contenders Line 9 of CPPL
        self.S_t = []

        # For the gradient descent formula in the literature,
        # for t = 0, we have to comply to the framework that a condorcet winner exists.
        self.winner_known = True

        # Termination flag
        self.is_finished = False

    def _init_checks(self) -> None:
        """Check whether folders exist for algorithm run and check the args values given by the user"""
        # Check Instance problem directory in arguments
        if self.args.directory != "No Problem Instance Directory given":
            self.get_problem_instances()
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
            parameter_directory = pathlib.Path(
                f"{Constants.PARAM_POOL_FOLDER.value}_{self.args.solver}"
            )
            if parameter_directory.exists():
                pass
            else:
                os.mkdir(f"{Constants.PARAM_POOL_FOLDER.value}_{self.args.solver}")

        # Check if training is required
        # As for now the training is not required, this was used for experimenting with ReACTR setup.
        # You can ignore this.
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
        """Get parameters from the solver's json file.

        The parameters of the solvers are in the json files in `solver_parameter_schemas/param_<solver>.json` format

        Returns
        -------
        parameters: Dict
            A dictionary of the parameters where the key is the argument of the solver and the values are the argument's possible values.
        """
        json_file_name = "params_" + str(self.args.solver)

        # Read the parameters.
        with open(
            f"{Constants.PARAMS_JSON_FOLDER.value}/{json_file_name}.json", "r"
        ) as file:
            data = file.read()
        parameters = json.loads(data)
        param_names = list(parameters.keys())  # for validation of the schema

        # Store the solver general solver parameter schema for checking the format
        with open(
            f"{Constants.PARAMS_JSON_FOLDER.value}/{Constants.PARAM_SCHEMA_JSON_FILE.value}",
            "r",
        ) as file:
            schema = file.read()
        schema_meta = json.loads(schema)

        # Validate whether the solver's parameters match the general schema
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
        """Initialize the parameter configuration pool or the arms"""

        # If there is no prior parameters given then create new ones.
        if self.args.data is None:
            pool_keys = [
                f"contender_{c}" for c in range(self.num_contenders)
            ]  # create the keys of the arms to keep track
            self.contender_pool = dict.fromkeys(pool_keys, 0)

            # Only used if comparing any other configurator.
            if self.args.baselineperf:
                print("Baseline Performance Run (only default parameters)")
                # Create the contenders based on the solver's parameters
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = set_genes(
                        solver_parameters=self.solver_parameters
                    )
                    self.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                    )
            else:
                # Get the random parameter set from thr solver's parameters
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = random_genes.get_genes_set(
                        solver=self.args.solver
                    )
                    self.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                    )

                if self.args.pws is not None:
                    contender_index = "contender_0"
                    self.contender_pool[contender_index] = set_genes(
                        solver_parameters=self.solver_parameters
                    )
                    self.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                    )  # Create the contenders pool in a directory
        elif self.args.data == "y":
            with open(f"{self.pool_filename}", "r") as file:
                self.contender_pool = eval(file.read())
                for contender_index in self.contender_pool:
                    self.set_contender_params(
                        contender_index=contender_index,
                        contender_pool=self.contender_pool[contender_index],
                    )

    def get_problem_instances(self) -> None:
        """Track the problem instances if they are provided by the user."""
        self.directory = os.fsencode(self.args.directory)
        # Get the sorted list of the problem instances based on the file ordering by the user.
        # The default ardering is in acending.
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

        # Track the problem instance names to be solved in problem_instance_list.txt
        with open(f"{Constants.PROBLEM_INSTANCE_LIST_TXT_FILE.value}", "w") as file:
            print(self.problem_instance_list, file=file)

    def _init_pca_features(self) -> Tuple[StandardScaler, int, PCA]:
        """Return the features of the problem instances after applying PCA to reduce the feature dimensionality.

        Returns
        -------
        standard_scalar: StandardScalar
            An object of the standard scalar class used for normalization.
        n_pca_features_components: int
            Number of the dimensions for the PCA of the instance features.
        pca_obj_instances: PCA
            Fitted PCA class object on the reduced instance features dimensions.
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
        features = self._read_training_features_csv()

        standard_scalar = StandardScaler()
        features = standard_scalar.fit_transform(features)

        # PCA on features
        n_pca_features_components = self.args.nc_pca_f
        pca_obj_instances = PCA(n_components=n_pca_features_components).fit(features)

        return standard_scalar, n_pca_features_components, pca_obj_instances

    def _read_training_features_csv(self) -> np.ndarray:
        """Extract the problem instance features from the csv file.

        Returns
        -------
        np.ndarray
            Problem instance features as a numpy array.
        """
        features = []
        with open(
            f"Instance_Features/training_features_{self.directory}.csv", "r"
        ) as csvFile:
            reader = csv.reader(csvFile)
            next(reader)
            for row in reader:
                if len(row[0]) != 0:
                    next_features = row
                    next_features.pop(0)
                    next_features = [float(j) for j in next_features]
                    features.append(next_features)
        csvFile.close()

        return np.asarray(features)

    def _init_pca_params(self) -> Tuple[np.ndarray, Dict, MinMaxScaler, int, PCA]:
        """Apply the PCA on the solver parameter's features.

        Returns
        -------
        params: np.ndarray
            A numpy array of the prameter's property of the given solver.
        parameter_value_dict: Dict
            A dictionary of the parameter with the key as the solver's parameter name and the value is the parameter value.
        min_max_scalar: MinMaxScaler
            A MinMaxScaler object.
        n_pca_params_components: int
            Number of the dimensions for the PCA on parameter features.
        pca_obj_params: PCA
            Fitted PCA class object on the reduced parameter features dimensions.
        """
        params, parameter_value_dict = self.cppl_utils.read_parameters()
        params = np.asarray(params)
        all_min, all_max = random_genes.get_all_min_and_max(self.solver_parameters)
        all_min, _ = self.cppl_utils.read_parameters(contender_genes=all_min)
        all_max, _ = self.cppl_utils.read_parameters(contender_genes=all_max)
        params = np.append(params, [all_min, all_max], axis=0)
        params = log_space_convert(
            limit_number=float(self.args.paramlimit),
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
        """Return the context vector dimensions based on the joint feature map mode.

        Returns
        -------
        candidates_pool_dimensions: int
            The context vector dimensions from the joint feature map mode. Corresponds to `d` in the literature.
        """
        self.joint_featured_map_mode = (
            self.args.jfm
        )  # Default="polynomial" used by the CPPL framework
        candidates_pool_dimensions = 4
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
        self, features, params, n_arms
    ) -> np.ndarray:
        """Extract and create the context matrix on the problem instance features and solver parameters.

        The context features are build by using the joint feature map on the instance features and solver's parameters features.
        The dimension of each context vector for each arm is `d` calculated based on the joint feature map mode given by the user.

        Parameters
        ----------
        features: np.ndarray
            The problem instance features.
        params: np.ndarray
            The parameters of the solver.
        n_amrs: int
            The number of parameter configurations in the candidate pool.

        Returns
        -------
        context_matrix: np.ndarray
            A 2-D matrix of shape (n_arms, d) for the given problem instance.
        """
        # PCA on parametrization
        params_transformed = self.pca_obj_params.transform(params)
        # construct X_t (context specific (instance information) feature matrix ( and parameterization information))
        context_vector_dimension = len(
            self.theta_bar
        )  # Distinct Parameters or available arms
        context_matrix = np.zeros((n_arms, context_vector_dimension))
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
        context_matrix = normalize(context_matrix, norm="max", copy=False)
        return context_matrix

    def get_skill_vector(self, n_arms: int, context_matrix: np.ndarray) -> np.ndarray:
        """Get the skill vector, i.e., the latent utility estimate based on the contextual information

        Parameters
        ----------
        n_arms : int
            The number of parameter configurations in the candidate pool.
        context_matrix : np.ndarray
            A 2-D matrix of shape (n_arms, d) for the given problem instance.

        Returns
        -------
        skill_vector: np.ndarray
            A vector containing the latent utility estimate of each arm based on the contextual information.
        """
        # compute estimated contextualized utility parameters (v_hat)
        skill_vector = np.zeros(n_arms)  # Line 7 in CPPL algorithm
        for i in range(n_arms):
            skill_vector[i] = np.exp(np.inner(self.theta_bar, context_matrix[i, :]))
        return skill_vector

    def get_parameterizations(self) -> np.ndarray:
        """Return the parameters of the solver.

        Returns
        -------
        params: np.ndarray
            Normalized parameters of the solver.
        """
        # get parametrization
        params, _ = self.cppl_utils.read_parameters()
        params = np.asarray(params)
        params = log_space_convert(
            limit_number=self.parameter_limit,
            param_set=params,
            solver_parameter=self.solver_parameters,
        )
        params = self.min_max_scalar.transform(params)
        return params

    def get_pca_instance_features(self, problem_instance_name: str) -> np.ndarray:
        """Get the reduced features of the problem instances.

        Parameters
        ----------
        problem_instance_name : str
            Name of the problem instance.

        Returns
        -------
        features: np.ndarray
            Instance features with reduced dimensionality.
        """
        # read and preprocess instance features (PCA)
        features = self.get_features(problem_instance_name=f"{problem_instance_name}")
        features = self.standard_scalar.transform(features.reshape(1, -1))
        features = self.pca_obj_instances.transform(features)
        return features

    def get_features(self, problem_instance_name: str) -> np.ndarray:
        """Get normal features of the problem instance from the csv file.

        Parameters
        ----------
        problem_instance_name : str
            Name of the problem instance.

        Returns
        -------
        np.ndarray
            Returns the normal features of the problem instance from the csv file in form of a numpy array.
        """
        with open(f"Instance_Features/Features_{self.directory}.csv", "r") as csvFile:
            reader = csv.reader(csvFile)
            next(reader)
            for row in reader:
                row_list = row[1:]
                features = [float(j) for j in row_list]
                if os.path.basename(row_list[0]) == problem_instance_name:
                    csvFile.close()
                    break
        return np.asarray(features)

    def set_contender_params(
        self,
        contender_index: int,
        contender_pool: Dict,
        return_it: bool = False,
    ) -> Optional[List]:
        """
        Set the parameter configuration for the contenders, i.e., arms.

        Parameters
        ----------
        contender_index : int
            Index of the contender for parameterization.
        contender_pool : Dict
            The pool of contenders participating in the tournament.
        return_it : bool, default=False
            Boolean whether to return the parameter set.

        Returns
        -------
        parameter_set: Optional[List]
            The parameter set of the contender ot arm.
        """
        param_names = list(self.solver_parameters.keys())
        params = self.solver_parameters

        parameter_set = [0 for _ in range(len(contender_pool))]

        for i in range(len(contender_pool)):
            if "flag" in params[param_names[i]]:
                parameter_set[i] = str(contender_pool[i])
            else:
                if self.solver_parameters[param_names[i]]["paramtype"] == "discrete":
                    parameter_set[i] = str(param_names[i]) + str(int(contender_pool[i]))
                else:
                    parameter_set[i] = str(param_names[i]) + str(contender_pool[i])

        with open(
            f"{Constants.PARAM_POOL_FOLDER.value}_{self.args.solver}/"
            + str(contender_index),
            "w",
        ) as file:
            print(" ".join(parameter_set), file=file)

        if return_it:
            return parameter_set
