import numpy as np
from scipy.linalg import sqrtm
from utils.utility_functions import hessian


class UCB:
    """The Upper Confidence Bound algorithm with Context."""
    def __init__(
        self,
        cppl_base_object: CPPLBase,
        context_matrix: np.ndarray,
        degree_of_freedom: int,  # degree of freedom (len of theta_bar)
        n_arms: int,  # Number of parameters
        v_hat: float,  # estimated unknown contextualized utility parameter
    ):
        
        self.base = cppl_base_object
        self.S_t = self.base.S_t
        self.context_matrix = context_matrix
        self.degree_of_freedom = degree_of_freedom
        self.n_arms = n_arms
        self.v_hat = v_hat

        self.subset_size = self.base.subset_size
        self.gradient = self.base.grad
        self.grad_op_sum = self.base.grad_op_sum
        self.hess_sum = self.base.hess_sum
        self.omega = self.base.omega
        self.theta_bar = self.base.theta_bar
        self.time_step = self.base.time_step
        self.confidence_t = None
        if self.base.time_step == 0:
            self.initial_step = True
        else:
            self.initial_step = False

    def run(self):
        if self.initial_step:
            self.step()
            contender_list_str = self._update_contender_list_str()
            return self.S_t, contender_list_str, self.v_hat
        else:
            self.step()
            contender_list_str = self._update_contender_list_str()
            return self.S_t, self.confidence_t, contender_list_str, self.v_hat

    def step(self):
        if self.time_step == 0:
            self.S_t = self.get_best_subset()
        else:
            self.confidence_t = np.zero(self.n_arms)
            hess = hessian(theta=self.theta_bar, subset_arms=self.S_t, context_matrix=self.context_matrix)
            self.hess_sum += hess
            self.grad_op_sum += np.outer(self.gradient, self.gradient)

            try:
                V_hat = np.asarray((1 / self.time_step) * self.grad_op_sum).astype(
                    "float64"
                )
                S_hat = np.asarray((1 / self.time_step) * self.hess_sum).astype(
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
                            + self.degree_of_freedom
                            + 2
                            * np.sqrt(self.degree_of_freedom * np.log(self.time_step))
                        )
                        * np.linalg.norm(Sigma_hat_sqrt * M_i * Sigma_hat_sqrt, ord=2)
                    )  # Equation of confidence bound in section 5.3 of https://arxiv.org/pdf/2002.04275.pdf
                self.S_t = self.get_best_subset(exception=False)
            except:
                self.S_t = self.get_best_subset()

            self.confidence_t = self.confidence_t / max(self.confidence_t)
            self.v_hat = self.v_hat / max(self.v_hat)

    def get_best_subset(self, exception=True):
        if exception:
            return (-self.v_hat).argsort()[0 : self.subset_size]
        else:
            return (-(self.v_hat + self.confidence_t)).argsort()[0 : self.subset_size]

    def _update_contender_list_str(self):
        contender_list_str = []
        for i in range(self.subset_size):
            contender_list_str.append("contender_" + str(self.S_t))
        return contender_list_str
