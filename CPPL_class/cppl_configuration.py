"""A configuration class for CPPL algorithm."""
import logging
import multiprocessing as mp
import random
from typing import List, Tuple, Union

import numpy as np

from CPPL_class.cppl_base import CPPLBase
from preselection import UCB
from utils import set_param, random_genes
from utils.log_params_utils import log_space_convert
from utils.utility_functions import set_genes, join_feature_map


class CPPLConfiguration:
    """A CPPL configuration class which handles the configuration functionality ofthe CPPL algorithm.

    Parameters
    ----------
    args : _type_
        _description_
    logger_name : str, optional
        _description_, by default "CPPLConfiguration"
    logger_level : _type_, optional
        _description_, by default logging.INFO
    """

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

    def _get_contender_list(self, filename: str) -> Tuple[np.ndarray, List[str], List]:
        """_summary_

        Parameters
        ----------
        filename : str
            _description_

        Returns
        -------
        Tuple[np.ndarray, List[str], List]
            _description_
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

    def _generate_new_parameters(self) -> List:
        """_summary_

        Returns
        -------
        List
            _description_
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
    def _parallel_evolution_and_fitness(self) -> Tuple[List, List]:
        """_summary_

        Returns
        -------
        Tuple[List, List]
            _description_
        """
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

    def _evolution_and_fitness(
        self, new_candidates_size: List
    ) -> Tuple[np.ndarray, List]:
        """_summary_

        Parameters
        ----------
        new_candidates_size : List
            _description_

        Returns
        -------
        Tuple[np.ndarray, List]
            _description_
        """
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

    def _contender_list_including_generated(self) -> List[str]:
        """_summary_

        Returns
        -------
        List[str]
            _description_
        """
        contender_list = []

        _, _, _, _, _, v_hat = self.base.get_context_feature_matrix(
            filename=self.filename
        )

        S_t = (-v_hat).argsort()[0 : self.base.subset_size]

        for index in range(self.base.subset_size):
            contender_list.append("contender_" + str(S_t[index]))

        return contender_list

    def update_logs(self, contender_index: Union[int, str], genes: List[str]) -> None:
        """_summary_

        Parameters
        ----------
        contender_index : Union[int, str]
            _description_
        genes : List[str]
            _description_
        """
        self.base.contender_pool[
            "contender_" + str(contender_index)
        ] = genes  # TODO: change to local pool after clarification
        self.base.tracking_pool.info(self.base.contender_pool)
