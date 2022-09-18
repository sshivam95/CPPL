from typing import Dict, List, Optional


def set_contender_params(
    contender_index: int,
    contender_pool: Dict,
    solver_parameters: Dict,
    return_it: bool = False,
) -> Optional[List]:
    """
    Set the parameters for the contenders, i.e., arms.

    Parameters
    ----------
    contender_index : int
        Index of the contender for parameterization.
    contender_pool : Dict
        The pool of contenders participating in the tournament.
    solver_parameters : Dict
        The parameter set used by the solver.
    return_it : bool, default=False
        Boolean whether to return the parameter set.

    Returns
    -------
    parameter_set: Optional[List]
        The parameter set of the contender ot arm.
    """
    param_names = list(solver_parameters.keys())
    params = solver_parameters

    parameter_set = [0 for _ in range(len(contender_pool))]

    for i in range(len(contender_pool)):
        if "flag" in params[param_names[i]]:
            parameter_set[i] = str(contender_pool[i])
        else:
            if solver_parameters[param_names[i]]["paramtype"] == "discrete":
                parameter_set[i] = str(param_names[i]) + str(int(contender_pool[i]))
            else:
                parameter_set[i] = str(param_names[i]) + str(contender_pool[i])

    with open("ParamPool/" + str(contender_index), "w") as file:
        print(" ".join(parameter_set), file=file)

    if return_it:
        return parameter_set
