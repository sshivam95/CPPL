import random
from typing import List, Tuple
import numpy as np
from utils import random_genes
from utils.log_params_utils import log_space_convert


def evolution_and_fitness(
    best_candidate,
    second_candidate,
    new_candidates,
    new_candidates_size: int,
    candidate_parameters_size,
    base,
) -> Tuple[np.ndarray, List]:
    """A single step to generate new parameters as contenders through genetic engineering approach.

    Parameters
    ----------
    new_candidates_size : int
        The number of newly generated parameters as contenders.

    Returns
    -------
    new_candidates_transformed : np.ndarray
        Newly generated candidate parameters transformed in the log space.
    new_candidates : List
        List of newly generated candidate parameters through one hot decode.
    """
    # Generation approach based on genetic mechanism with mutation and random individuals
    new_candidates = np.zeros(
        shape=(
            new_candidates_size,
            new_candidates,
        )  # TODO: The second shape can be shanged to candidate_parameters_size after clearing the doubt.
    )

    for candidate in range(new_candidates_size):
        random_individual = random.uniform(0, 1)
        next_candidate = np.zeros(candidate_parameters_size)
        contender = random_genes.get_genes_set(
            solver=base.args.solver,
            solver_parameters=base.solver_parameters,
        )
        genes, _ = base.cppl_utils.read_parameters(contender_genes=contender)
        mutation_genes = random_genes.get_one_hot_decoded_param_set(
            genes=genes,
            solver=base.args.solver,
            param_value_dict=base.parameter_value_dict,
            solver_parameters=base.solver_parameters,
        )

        for index in range(candidate_parameters_size):
            random_seed = random.uniform(0, 1)
            mutation_seed = random.uniform(0, 1)

            # Dueling function
            if random_seed > 0.5:
                next_candidate[index] = best_candidate[index]
            else:
                next_candidate[index] = second_candidate[candidate]
            if mutation_seed < 0.1:
                next_candidate[index] = mutation_genes[index]

        if random_individual < 0.99:
            new_candidates[candidate] = mutation_genes
        else:
            new_candidates[candidate] = next_candidate

    new_candidates = random_genes.get_one_hot_decoded_param_set(
        genes=new_candidates,
        solver=base.args.solver,
        solver_parameters=base.solver_parameters,
        reverse=True,
    )
    new_candidates_transformed = log_space_convert(
        limit_number=base.parameter_limit,
        param_set=new_candidates,
        solver_parameter=base.solver_parameters,
    )
    return new_candidates_transformed, new_candidates
