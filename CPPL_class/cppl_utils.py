"""Utility functions in the CPPL algorithm."""
import logging
from typing import Dict, List, Tuple

import numpy as np

import utils.utility_functions


class CPPLUtils:
    """A class containing utility functions used in the CPPL algorithm.

    Parameters
    ----------
    pool : Dict[str, int]
        The pool of contenders or parameters to solve the problem instance.
    solver : str
        Solver used to solve the instances.
    solver_parameters : Dict
        The parameter set used by the solver.
    logger_name : str, optional
        Name of the logger, by default "CPPLUtils"
    logger_level : int, optional
        Level of the logger, by default logging.INFO
    """

    def __init__(
        self,
        pool: Dict[str, int],
        solver: str,
        solver_parameters: Dict,
        logger_name: str = "CPPLUtils",
        logger_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.pool = pool
        self.solver = solver
        self.solver_parameters = solver_parameters

    def read_parameters(self, contender_genes: List[int] = None) -> Tuple[np.ndarray, Dict]:
        """Return the parameter names and the actual parameters of the given contender.

        Parameters
        ----------
        contender_genes : List[int], optional
            The genes of the contender parameter , by default None

        Returns
        -------
        parameters_name_list : np.ndarray
            A numpy array of the prameter's property of the given solver
        parameter_value_dict : Dict
            A dictionary of the parameter with the key as the solver's parameter name and the value is the parameter value.
        """
        global parameter_value_dict
        parameter_names, params = utils.utility_functions.get_solver_params(
            solver_parameters=self.solver_parameters, solver=self.solver
        )

        if contender_genes is not None:
            parameters_name_list, parameter_value_dict = self.read_param_from_dict(
                contender_genes=contender_genes, parameter_names=parameter_names
            )
            return np.asarray(parameters_name_list), parameter_value_dict

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
                parameters_name_list, parameter_value_dict = self.read_param_from_dict(
                    contender_genes=new_params_list[new_contender],
                    parameter_names=parameter_names,
                )
                new_params_list.append(parameters_name_list)
            return np.asarray(new_params_list), parameter_value_dict

    def read_param_from_dict(
        self, contender_genes: List[int], parameter_names: List
    ) -> Tuple[List[int], Dict]:
        """Get the parameter set from the solver's parameter.

        Parameters
        ----------
        contender : List[int]
            The genes of the contender parameter
        parameter_names : List
            List of solver's parameters names.

        Returns
        -------
        parameters_name_list : List[int]
            List of the property's names from the solver's parameters.
        parameter_value_dict : Dict
            A dictionary of the parameter with the key as the solver's parameter name and the value is the parameter value.
        """
        global index
        next_params = contender_genes
        solver_parameters = self.solver_parameters
        originals_index_to_delete = []
        one_hot_addition = []
        parameter_value_dict = {}

        if len(parameter_names) == len(contender_genes):
            for i, _ in enumerate(parameter_names):
                parameter_value_dict[parameter_names[i]] = next_params[i]

                if solver_parameters[parameter_names[i]]["paramtype"] == "categorical":
                    if solver_parameters[parameter_names[i]]["valtype"] == "int":
                        min_value = solver_parameters[parameter_names[i]]["minval"]
                        max_value = solver_parameters[parameter_names[i]]["maxval"]
                        value_range = max_value - min_value + 1
                        values = [min_value + j for j in range(value_range)]
                        index = int(next_params[i]) - min_value

                    else:  # params[parameter_names[i]]["valtype"] == "str"
                        values = solver_parameters[parameter_names[i]]["values"]

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

            parameters_name_list = []

            for key in parameter_value_dict:
                if parameter_value_dict[key] is not None:
                    parameters_name_list.append(parameter_value_dict[key])

            parameters_name_list += one_hot_addition

        else:
            parameters_name_list = contender_genes

        return parameters_name_list, parameter_value_dict
