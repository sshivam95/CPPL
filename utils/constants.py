"""Constants used in the project."""
from enum import Enum


class Constants(Enum):
    """Constants class to track all the constants used in the code for files and folders."""

    # Files and folders
    PARAMS_JSON_FOLDER = "solver_parameter_schemas"  # Folder which contains all the parameters schema for the solvers
    PARAM_SCHEMA_JSON_FILE = f"param_schema.json"  # Parameter schema for validation
    PROBLEM_INSTANCE_LIST_TXT_FILE = (
        "problem_instance_list.txt"  # File to track all the instances to solve
    )
    INSTANCE_FEATURES_FOLDER = (
        "Instance_Features"  # Folder which contains all the instances
    )
    PARAM_POOL_FOLDER = "contender_pool"  # Folder containing all the contenders
    INSTANCE_FEATURES_PATH = (
        f"{INSTANCE_FEATURES_FOLDER}/training_features_"  # Training instance path
    )
