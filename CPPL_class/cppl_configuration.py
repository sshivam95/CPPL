"""A configuration class for CPPL algorithm."""
from argparse import Namespace
import logging
import multiprocessing as mp
import random
import sys
from typing import List, Tuple, Union

import numpy as np

from CPPL_class.cppl_base import CPPLBase
from preselection import UCB
from utils import set_param, random_genes
from utils.genetic_parameterization import evolution_and_fitness
from utils.log_params_utils import log_space_convert
from utils.utility_functions import set_genes, join_feature_map


class CPPLConfiguration:
    """A CPPL configuration class which handles the configuration functionality ofthe CPPL algorithm.

    Parameters
    ----------
    args : Namespace
        The arguments of the algorithm given by user.
    logger_name : str, optional
        Logger name, by default "CPPLConfiguration"
    logger_level : int, optional
        Level of the logger, by default logging.INFO
        
    Attributes
    ----------
    filename : str
        The name of the problem instance file.
    async_results : List
        The results of the asynchronous processes.
    n_arms : int
        Number of contenders or arms in the contender pool, by default None.
    params : np.ndarray
        A parameter array of all the contenders in the pool, by default None.
    pca_context_features : np.ndarray
        A numpy array of transformed context features, by default None.
    discard: List 
        A list of discarded contenders, by default None.
    subset_contender_list_str: List[str] 
        A list of the subset of arms from the pool which contains `CPPLBase.subset_size` (=number of cores in the CPU) arms with the highest upper bounds on the latent utility, by default None.
    new_candidates_size : int
        Number of new generating candidates through genetic selection, by default 1000.
    best_candidate 
        The best candidate in the subset with the highest upper bounds on the latent utility, by default None.
    second_candidate 
        The second best candidate in the subset, by default None.
    candidate_parameters_size: int 
        Length of the parameter array of the candidate, by default None.
    """

    def __init__(
        self,
        args: Namespace,
        logger_name: str = "CPPLConfiguration",
        logger_level: int = logging.INFO,
    ):
        self.base = CPPLBase(args=args)  # Create a base class object.
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.filename = None
        self.async_results = []
        self.n_arms: int = None
        self.params: np.ndarray = None
        self.pca_context_features = None
        self.discard: List = None
        self.subset_contender_list_str: List[str] = None
        self.new_candidates_size = 1000
        self.best_candidate = None
        self.second_candidate = None
        self.candidate_parameters_size: int = None

    def _get_contender_list(self, filename: str) -> Tuple[np.ndarray, List[str], List]:
        """Returns the subset of contenders from the pool with the highest upper bounds on the latent utility.

        Parameters
        ----------
        filename : str
            Name of the Problem instance file.

        Returns
        -------
        context_matrix : np.ndarray
            A context matrix where each element is associated with one of the different arms and contains
            the properties of the arm itself as well as the context in which the arm needs to be chosen.
        subset_contender_list_str : List[str]
            A list of the subset of arms from the pool which contains `CPPLBase.subset_size` (=number of cores in the CPU) arms with the highest upper bounds on the latent utility.
        discard : List
            A list of discarded contenders.
        """
        self.filename = filename
        (
            context_matrix,  # Context matrix
            context_vector_dimension,  # context vector dimension (len of theta_bar)
            self.pca_context_features,
            self.n_arms,  # Number of parameters
            self.params,
            v_hat,  # Estimated skill parameter
        ) = self.base.get_context_feature_matrix(filename=self.filename)
        self.discard = []

        ucb = UCB(
            cppl_base_object=self.base,
            context_matrix=context_matrix,
            context_vector_dimension=context_vector_dimension,
            n_arms=self.n_arms,
            v_hat=v_hat,
        )
        if self.base.time_step == 0:
            (
                self.base.S_t,
                self.subset_contender_list_str,
                v_hat,
            ) = ucb.run()  # Get the subset S_t
            
            print(f"Subset of contenders from pool: {self.base.S_t}")
            
            if self.base.args.exp == "y":
                for i in range(self.base.subset_size):
                    self.subset_contender_list_str[i] = f"contender_{i}"

            genes = set_genes(solver_parameters=self.base.solver_parameters)
            set_param.set_contender_params(
                contender_index="contender_0",
                contender_pool=genes,
                solver_parameters=self.base.solver_parameters,
            )
            self.subset_contender_list_str[0] = "contender_0"
            self.update_logs(winning_contender_index="0", genes=genes)

        else:
            # compute confidence_t and select S_t (symmetric group on [num_parameters], consisting of rankings: r ∈ S_n)
            (
                self.base.S_t,
                confidence_t,
                self.subset_contender_list_str,
                v_hat,
            ) = (
                ucb.run()
            )  # Get the subset S_t along with the confidence intervals if it is not the initial time step.

            print(f"Subset of contenders from pool: {self.base.S_t}")
            
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
                self.subset_contender_list_str = self._generate_new_parameters()

        return context_matrix, self.subset_contender_list_str, self.discard

    def _generate_new_parameters(self) -> List:
        """Generate new parameters for the contenders after discarding parameters from the discard list.

        Returns
        -------
        new_contender_list : List
            The new parameter list which are used as contenders.
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

        # for index in range(self.new_candidates_size):
            # print(f"Shape of new_candidates_transformed[{index}] : {new_candidates_transformed[index].shape}")
            # print(f"Shape of self.pca_context_features : {self.pca_context_features[0].shape}")
        print(f"Shape of new_candidates_transformed : {new_candidates_transformed.shape}")
        print(f"Shape of new_candidates_size : {self.new_candidates_size}")
            # context_vector = join_feature_map(
            #     x=new_candidates_transformed[index],
            #     y=self.pca_context_features[0],
            #     mode=self.base.joint_featured_map_mode,
            # )

            # v_hat_new_candidates[index] = np.exp(
            #     np.inner(self.base.theta_bar, context_vector)
            # )
        sys.exit(0)

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
            self.update_logs(winning_contender_index=self.discard[index], genes=genes)

        new_contender_list = self._contender_list_including_generated()

        return new_contender_list

    # TODO convert it into a class
    def _parallel_evolution_and_fitness(self) -> Tuple[List, List]:
        """Generate new parameters as contenders parallely through genetic engineering approach.

        Returns
        -------
        new_candidates_transformed : List
            List of newly generated candidate parameters transformed in the log space.
        new_candidates : List
            List of newly generated candidate parameters through one hot decode.
        """
        candidate_pameters = random_genes.get_one_hot_decoded_param_set(
            genes=self.params[self.base.S_t[0]],
            solver=self.base.args.solver,
            param_value_dict=self.base.parameter_value_dict,
            solver_parameters=self.base.solver_parameters,
        )
        self.candidate_parameters_size = len(candidate_pameters)
        self.new_candidates = np.zeros(
            shape=(self.new_candidates_size, self.candidate_parameters_size)
        )  # TODO: Remove this after clearning doubt.

        last_step = self.new_candidates_size % self.base.subset_size
        new_candidates_size = self.new_candidates_size - last_step  # TODO: Check this if the new candidate size have to change or not after clearing doubt.
        step_size = (new_candidates_size) / self.base.subset_size
        all_steps = []

        for _ in range(self.base.subset_size):
            all_steps.append(int(step_size))
        all_steps.append(int(last_step))

        step = 0
        pool = mp.Pool(processes=self.base.subset_size)

        print(f"all_steps: {all_steps}")
        for index, _ in enumerate(all_steps):
            step += all_steps[index]
            pool.apply_async(
                evolution_and_fitness,
                args=(self.best_candidate, self.second_candidate, len(self.new_candidates[0]), all_steps[index], self.candidate_parameters_size, self.base),
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

    def save_result(self, result:Tuple[np.ndarray, List]) -> None:
        """Save the results of asynchronous processes.

        Parameters
        ----------
        result : Tuple[np.ndarray, List]
            A tuple consisting of new candidates and their values in the log space.
        """
        new_candidates_transformed = result[0]
        new_candidates = result[1]
        self.async_results.append([new_candidates_transformed, new_candidates])


    def _contender_list_including_generated(self) -> List[str]:
        """Returns contenders from the subset with newly generated one through genetic approach.

        Returns
        -------
        contender_list : List[str]
            A contenders list from the subset with newly generated one through genetic approach.
        """
        contender_list = []

        _, _, _, _, _, v_hat = self.base.get_context_feature_matrix(
            filename=self.filename
        )

        S_t = (-v_hat).argsort()[0 : self.base.subset_size]

        for index in range(self.base.subset_size):
            contender_list.append("contender_" + str(S_t[index]))

        return contender_list

    def update_logs(self, winning_contender_index: Union[int, str], genes: List[str]) -> None:
        """Update the pool with the winning parameter and the logs to keep track of the contenders in the pool.

        Parameters
        ----------
        winning_contender_index : Union[int, str]
            Index of the winner contender
        genes : List[str]
            The winning parameters in the tournament.
        """
        self.base.contender_pool[
            "contender_" + str(winning_contender_index)
        ] = genes  # TODO: change to local pool after clarification
        self.base.tracking_pool.info(self.base.contender_pool)
