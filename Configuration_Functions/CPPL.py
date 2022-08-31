"""A CPPL class which represents CPPL algorithm"""
import csv
import json
import os
import pathlib
import sys
import logging
import multiprocessing as mp
import jsonschema
import numpy as np

from sklearn.preprocessing import StandardScaler

from Configuration_Functions import file_logging, set_param, random_genes
from Configuration_Functions.Constants import Constants


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


class CPPL(object):
    """A generic CPPL Class which includes initialization, running and updating the parameters in CPPL algorithm.

    Parameters
    ----------
    args: Namespace
        The namespace variable for all the arguments in the parser.
    """

    def __int__(self, args, logger_name='Configuration_Functions.CPPL', logger_level=logging.INFO):
        self.args = args
        self.num_parameter = mp.cpu_count()  # Total number of Parameters in set P
        self.features = list()
        self.train_list = list()

        # Creating tracking logs
        self.tracking_times = file_logging.tracking_files(
            filename=self.args.times_file_name,
            logger_name=logger_name,
            level=logger_level
        )
        self.tracking_pool = file_logging.tracking_files(
            filename=Constants.POOL_JSON_FILE.value,
            logger_name=logger_name,
            level=logger_level
        )
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        # Initialization
        self._check_arguments()
        self.solver_parameters = self._get_solver_parameters()
        self._init_pool()

        self.tracking_pool.info(self.contender_pool)  # Write contender_pool to textfile for solver to access parameter settings

        # initialize features from instance files
        self._init_features()

        # TODO add other initialization methods

    def _check_arguments(self):

        # Check Instance problem directory in arguments
        if self.args.directory != Constants.NO_INSTANCE_DIRECTORY_FOUND.value:
            self._init_output_files()
        else:
            print(Constants.NO_DIRECTORY_MESSAGE.value)
            sys.exit(0)

        # Check if the Instance feature directory available
        self._check_instance_feature_directory()

        # Check Solver in arguments
        if self.args.solver == Constants.NO_SOLVER_FOUND.value:
            print("\nYou need to choose a solver!!!\n\n**[-s <solver_name>]**\n\n")
            sys.exit(0)
        else:
            parameter_directory = pathlib.Path(f'{Constants.PARAM_POOL_FOLDER.value}')
            if parameter_directory.exists():
                pass
            else:
                os.mkdir(f'{Constants.PARAM_POOL_FOLDER.value}')

        # Check training is required
        if self.args.train_number is not None:
            for self.root, self.training_directory, self.training_file in sorted(os.walk(self.directory)):
                continue

        if self.args.train_rounds is not None:
            self.rounds_to_train = int(self.args.train_rounds)
        else:
            self.rounds_to_train = 0

    def _init_output_files(self):
        self.directory = os.fsencode(self.args.directory)
        # path, dirs, files = next(os.walk(self.args.directory))
        self.instances = []
        if self.args.file_order == Constants.ARGS_ASCENDING.value:
            self.problem_instance_list = get_problem_instance_list(
                sorted_directory=sorted(os.listdir(self.directory)))
        elif self.args.file_order == Constants.ARGS_DESCENDING.value:
            self.problem_instance_list = get_problem_instance_list(sorted(os.listdir(self.directory),
                                                                          reverse=True))
        else:
            file_order = str(self.args.file_order)
            with open(f"{file_order}.txt",
                      "r") as file:
                self.problem_instance_list = eval(file.read())
        with open(f"{Constants.PROBLEM_INSTANCE_LIST_TXT_FILE.value}",
                  "w") as file:
            print(self.problem_instance_list,
                  file=file)  # Print all the instance in problem_instance_list.txt

    def _get_solver_parameters(self):
        json_file_name = "params_" + str(self.args.solver)

        with open(f"Configuration_Functions/{json_file_name}.json",
                  "r") as file:
            data = file.read()
        parameters = json.loads(data)
        param_names = list(parameters.keys())

        with open(f'{Constants.PARAM_SCHEMA_JSON_FILE.value}',
                  "r") as file:
            schema = file.read()
        schema_meta = json.loads(schema)

        for parameter_name in param_names:
            valid = json_validation(param=parameters[parameter_name],
                                    schema=schema_meta)
            if not valid:
                print("Invalid JSON data structure. Exiting.\n\n")
                print(f'{parameters[parameter_name]}')
                sys.exit(0)
        return parameters

    def _init_pool(self):
        if self.args.data is None:
            pool_keys = [f'contender_{c}' for c in range(self.args.contenders)]
            self.contender_pool = dict.fromkeys(pool_keys,
                                                0)

            if self.args.baselineperf:
                print("Baseline Performance Run (onl%s default parameters)")
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = set_genes(solver_parameters=self.solver_parameters)
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        genes=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters
                    )
            else:
                for contender_index in self.contender_pool:
                    self.contender_pool[contender_index] = random_genes.genes_set(solver=self.args.solver)
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        genes=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters,
                    )

                if self.args.pws is not None:
                    contender_index = "contender_0"
                    self.contender_pool[contender_index] = set_genes(solver_parameters=self.solver_parameters)
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        genes=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters
                    )
        elif self.args.data == Constants.ARGS_YES.value:
            pool_file = Constants.POOL_JSON_FILE.value

            if self.args.exp == Constants.ARGS_YES.value:
                pool_file = f"Pool_exp_{self.args.solver}.json"
            with open(f"{pool_file}",
                      "r") as file:
                self.contender_pool = eval(file.read())
                for contender_index in self.contender_pool:
                    set_param.set_contender_params(
                        contender_index=contender_index,
                        genes=self.contender_pool[contender_index],
                        solver_parameters=self.solver_parameters)

    def _check_instance_feature_directory(self):
        instance_feature_directory = pathlib.Path("Instance_Features")
        if instance_feature_directory.exists():
            pass
        else:
            print(
                "\nWARNING!\n\nA directory named <Instance_Features> with a "
                ".csv file containing the instance features is necessary!"
            )

            if self.args.directory != Constants.NO_INSTANCE_DIRECTORY_FOUND.value:
                print("\nIt must be named: Features_" + str(self.directory)[2:-1] + ".csv")

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

    def _init_features(self):
        if os.path.isfile("Instance_Features/training_features_" + str(self.directory)[2:-1] + ".csv"):
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
        self._read_features_csv()

        self.standard_scalar = StandardScaler()
        self.features = self.standard_scalar.fit_transform(np.asarray(self.features))

    def _read_features_csv(self):
        with open(f"Instance_Features/training_features_{self.directory}.csv",
                  "r") as csvFile:
            reader = csv.reader(csvFile)
            next(reader)
            for row in reader:
                if len(row[0]) != 0:
                    next_features = row
                    self.train_list.append(row[0])
                    next_features.pop(0)
                    next_features = [float(j) for j in next_features]
                    self.features.append(next_features)
        csvFile.close()
