from enum import Enum


class Constants(Enum):
    # args constants
    ARGS_PWS = "pws"
    ARGS_SOLVER_CPLEX = "cplex"
    ARGS_SOLVER_GLUCOSE = "glucose"
    ARGS_SOLVER_CADICAL = "cadical"
    ARGS_YES = "y"
    ARGS_DESCENDING = "descending"
    ARGS_ASCENDING = "ascending"
    ARGS_POLYNOMIAL = "polynomial"
    ARGS_KRONECKER = "kronecker"
    ARGS_CONCATENATION = "concatenation"

    # Folders and files
    POOL_JSON_FILE = "Pool.json"
    PARAM_SCHEMA_JSON_FILE = "Configuration_Functions/paramSchema.json"
    PROBLEM_INSTANCE_LIST_TXT_FILE = "problem_instance_list.txt"
    INSTANCE_FEATURES_FOLDER = "Instance_Features"
    PARAM_POOL_FOLDER = "ParamPool"
    INSTANCE_FEATURES_PATH = "Instance_Features/training_features_"

    # Warning messages
    FEATURES_IS_NECESSARY_MESSAGE = "\nWARNING!\n\nA directory named <Instance_Features> with a .csv file containing the instance features is necessary!"
    NO_DIRECTORY_MESSAGE = "\n\nYou need to specify a directory containing the problem instances!\n\n**[-d directory_name]**\n\n"
    EXITING_MESSAGE = "\nExiting..."
    INSTANCE_FEATURE_VALUE_EXAMPLE_MESSAGE = "\ninstance1 feature_value1 feature_value2 " \
                                             "....\ninstance2 feature_value1..."
    DIRECTORY_NAME_CSV_MESSAGE = "\nIt must be named: ""Features_<problem_instance_directory_name>.csv"

    # Checks
    NO_SOLVER_FOUND = "No solver chosen"
    NO_INSTANCE_DIRECTORY_FOUND = "No Problem Instance Directory given"

