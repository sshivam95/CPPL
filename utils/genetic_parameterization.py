import random
from typing import List, Tuple
import numpy as np
from CPPL_class.cppl_base import CPPLBase
from utils import random_genes
from utils.log_params_utils import log_space_convert

def evolution_and_fitness(
    best_candidate: np.ndarray,
    second_candidate: np.ndarray,
    num_parameters: int,
    new_candidates_size: int,
    params_length: int,
    base: CPPLBase,
) -> Tuple[np.ndarray, List]:
    """Generate new parameters as contenders through genetic engineering approach.
       
    Parameters
    ----------
    best_candidate : np.ndarray
        The random parameters based on the best candidate in the subset arm. 
    second_candidate : np.ndarray
        The random parameters based on the second best candidate in the subset arm. 
    num_parameters : int
        The number of newly generated parameters as contenders.
    new_candidates_size : int
        The total number of candidates to be created in a single run.
    params_length : int
        Length of the random parameters based on the best candidate in the subset arm. 
    base : CPPLBase
        The base class object.

    Returns
    -------
    new_candidates_transformed : np.ndarray
        Newly generated candidate parameters transformed in the log space.
    num_parameters : List
        List of newly generated candidate parameters through one hot decode.

    """
    solver = base.args.solver
    solver_parameters = base.solver_parameters
    # Generation approach based on genetic mechanism with mutation and random individuals
    new_candidates = np.zeros(
        shape=(
            new_candidates_size,
            num_parameters,
        )
    )

    for candidate in range(new_candidates_size):
        random_individual = random.uniform(0, 1)
        next_candidate = np.zeros(params_length)
        contender = random_genes.get_genes_set(
            solver=solver,
            solver_parameters=solver_parameters,
        )
        genes, _ = base.cppl_utils.read_parameters(contender_genes=contender)
        mutation_genes = random_genes.get_one_hot_decoded_param_set(
            genes=genes,
            solver=solver,
            param_value_dict=base.parameter_value_dict,
            solver_parameters=solver_parameters,
        )

        for index in range(params_length):
            random_seed = random.uniform(0, 1)
            mutation_seed = random.uniform(0, 1)

            # Dueling function
            if random_seed > 0.5:
                next_candidate[index] = best_candidate[index]
            else:
                next_candidate[index] = second_candidate[index]
            if mutation_seed < 0.1:
                next_candidate[index] = mutation_genes[index]

        if random_individual < 0.99:
            new_candidates[candidate] = mutation_genes
        else:
            new_candidates[candidate] = next_candidate

    new_candidates = random_genes.get_one_hot_decoded_param_set(
        genes=new_candidates,
        solver=solver,
        solver_parameters=solver_parameters,
        reverse=True,
    )
    new_candidates_transformed = log_space_convert(
        limit_number=base.parameter_limit,
        param_set=new_candidates,
        solver_parameter=solver_parameters,
    )
    return new_candidates_transformed, new_candidates
