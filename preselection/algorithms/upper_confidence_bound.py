import logging
from typing import List, Tuple, Union

import numpy as np
from CPPL_class.cppl_base import CPPLBase
from scipy.linalg import sqrtm
from preselection.algorithms.preselection_base import Preselection
from utils.utility_functions import hessian


class UCB(Preselection):
    """The Upper Confidence Bound algorithm with Context.

    Parameters
    ----------
    cppl_base_object : CPPLBase
        CPPL base class object.
    context_matrix : np.ndarray
        A context matrix where each element is a context vector for a arm. Each vector encodes features of the context in which a arm must be chosen.
    context_vector_dimension : int
        It is defined as the dimension of the joint feature map vector on the context features.
    n_arms : int
        Total number of arms in the pool.
    v_hat : np.ndarray
        A matrix where each row represents estimated contextualized utility parameters of each arm in the pool.

    Attributes
    ----------
    subset_size : int
        The size of the subset of arms from the pool.
    gradient : np.ndarray
        A numpy array of size n_arms which contains the gradient of the log-likelihood function in the partial winner feedback scenario for each arm in the pool.
    hess_sum : np.ndarray
        The sum of the hessian matrix of the log-likelihood function in the partial winner feedback scenario over the time step.
    omega : float
        A hyper-parameter which helps to determine the confidence intervals.
    theta_bar : np.ndarray
        An array where each element is the mean of the estimated score parameters (theta_hat) over previous time steps. Initially equal to the estimated score parameters.
    time_step : int
        The cuurent time step of the algorithm.
    confidence_t : np.array, default=None
        An array of size equal to `n_arms` where each elements contains the confidence interval of each arm in the pool.
    initial_step : bool
        A boolean value indicating if the current time step is the initial one.
    """

    def __init__(
        self,
        cppl_base_object: CPPLBase,
        context_matrix: np.ndarray,
        context_vector_dimension: int,  # degree of freedom (len of theta_bar)
        n_arms: int,  # Number of parameters
        skill_vector: np.ndarray,  # mean observed rewards
        logger_name: str = "UCB",
        logger_level: int = logging.INFO,
    ) -> None:
        super().__init__(
            cppl_base_object=cppl_base_object,
            context_matrix=context_matrix,
            n_arms=n_arms,
            skill_vector=skill_vector,
        )

        self.context_vector_dimension = context_vector_dimension
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        
        if self.base.time_step == 0:
            self.initial_step = True
        else:
            self.initial_step = False

    def run(
        self,
    ) -> Union[
        Tuple[np.ndarray, List[str], np.ndarray],
        Tuple[np.ndarray, np.ndarray, List[str], np.ndarray],
    ]:
        """Run the algorithm until completion.

        Returns
        -------
        S_t : np.ndarray
            The subset of arms from the pool which contains `subset_size` arms with the highest upper bounds on the latent utility.
        confidence_t : np.ndarray
            If the time_step is not the initial time step, returns the confidence bounds of each arm in the pool.
        contender_list_str : List[str]
            A list containing the arms in the subset.
        v_hat : np.ndarray
            An updated matrix where each row represents estimated contextualized utility parameters of each arm in the pool.
        """
        self.step()
        contender_list_str = self.update_contender_list_str()
        self.base.regret[self.time_step] = self.compute_regret(
            theta=self.theta_bar, context_matrix=self.context_matrix, subset=self.S_t
        )
        if self.initial_step:
            return self.S_t, contender_list_str, self.skill_vector
        else:
            return self.S_t, self.confidence_t, contender_list_str, self.skill_vector

    def step(self) -> None:
        """Run one step of the algorithm."""
        print("Time step: ", self.time_step)
        # print(f"Indices of non zero elements in Preference matrix/ Grad op sum: \n {np.nonzero(self.base.grad_op_sum)} \nSize of Grad op sum: {self.base.grad_op_sum.shape}")
        if self.time_step == 0:
            self.S_t = self.get_best_subset()
        else:
            self.confidence_t = np.zeros(self.n_arms)
            hess = hessian(
                theta=self.theta_bar,
                subset_arms=self.S_t,
                context_matrix=self.context_matrix,
            )
            self.base.hess_sum += hess
            self.base.grad_op_sum += np.outer(
                self.gradient, self.gradient
            )  # Can be treated as the preference matrix

            try:
                V_hat = np.asarray((1 / self.time_step) * self.base.grad_op_sum).astype(
                    "float64"
                )
                S_hat = np.asarray((1 / self.time_step) * self.base.hess_sum).astype(
                    "float64"
                )
                S_hat_inv = np.linalg.inv(S_hat).astype("float64")
                Sigma_hat = (1 / self.time_step) * S_hat_inv * V_hat * S_hat_inv
                Sigma_hat_sqrt = sqrtm(Sigma_hat)

                for i in range(self.n_arms):
                    M_i = np.exp(
                        2 * np.dot(self.theta_bar, self.context_matrix[i, :])
                    ) * np.dot(
                        self.context_matrix[i, :], self.context_matrix[i, :]
                    )  # M_t ^(i) (theta)
                    self.confidence_t[i] = self.omega * np.sqrt(
                        (
                            2 * np.log(self.time_step)
                            + self.context_vector_dimension
                            + 2
                            * np.sqrt(
                                self.context_vector_dimension * np.log(self.time_step)
                            )
                        )
                        * np.linalg.norm(Sigma_hat_sqrt * M_i * Sigma_hat_sqrt, ord=2)
                    )  # Equation of confidence bound in section 5.3 of https://arxiv.org/pdf/2002.04275.pdf
                self.S_t = self.get_best_subset(exception=False)
            except:
                self.S_t = self.get_best_subset()

            self.confidence_t = self.confidence_t / max(self.confidence_t)
            self.skill_vector = self.skill_vector / max(self.skill_vector)

    def get_best_subset(self, exception=True) -> np.ndarray:
        """Returns the subset of arms from the pool which contains `subset_size` arms with the highest upper bounds on the latent utility.

        Parameters
        ----------
        exception : bool, optional
            If there is an exception while calculating the values of the confidence bounds and estimated utility, by default True

        Returns
        -------
        best_subset : np.ndarray
            The subset of arms from the pool which contains `subset_size` arms with the highest upper bounds on the latent utility.
        """
        best_subset = None
        if exception:
            best_subset = (-self.skill_vector).argsort()[0 : self.subset_size]
            return best_subset
        else:
            best_subset = (-(self.skill_vector + self.confidence_t)).argsort()[
                0 : self.subset_size
            ]
            return best_subset

    def update_contender_list_str(self) -> List[str]:
        """Returns a list containing the arms in the subset.

        Returns
        -------
        contender_list_str : List[str]
            A string list containinig the arms in the best subset.
        """
        contender_list_str = []
        for i in range(self.subset_size):
            contender_list_str.append("contender_" + str(self.S_t[i]))
        return contender_list_str

    def compute_regret(
        self, theta: np.ndarray, context_matrix: np.ndarray, subset: np.ndarray
    ) -> float:
        """Compute the regret.

        Parameters
        ----------
        theta : np.ndarray
            The score parameter of the arms.
        context_matrix : np.ndarray
            A context matrix where each element is a context vector for a arm. Each vector encodes features of the context in which a arm must be chosen.
        S : np.ndarray
            The subset of arms from the pool which contains `subset_size` arms with the highest upper bounds on the latent utility.

        Returns
        -------
        float
            The regret.
        """
        context_matrix = np.array(context_matrix)
        # compute v^* of all arms
        skill_vector = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            skill_vector[i] = np.exp(np.dot(theta, context_matrix[i]))

        # get best arm
        best_arm = np.argmax(skill_vector)
        if best_arm in subset:
            return 0
        else:
            return (
                skill_vector[best_arm] - np.max(skill_vector[subset])
            ) / skill_vector[best_arm]
