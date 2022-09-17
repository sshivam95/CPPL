import logging

import numpy as np

import utils.utility_functions


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
