from typing import List
from CPPL_class.cppl_base import CPPLBase
from preselection.model.plackett_luce_model import PlackettLuceModel


class Preselection:
    def __init__(
        self,
        cppl_base_object: CPPLBase,
        context_matrix,
        n_arms,
        skill_vector,
    ) -> None:
        self.base = cppl_base_object
        self.context_matrix = context_matrix
        self.n_arms = n_arms
        self.skill_vector = skill_vector

        # self.feedback_mechanism = PlackettLuceModel(
        #     num_arms=self.n_arms, skill_vector=self.skill_vector
        # )
        self.S_t = self.base.S_t
        self.subset_size = self.base.subset_size
        self.gradient = self.base.grad
        self.omega = self.base.omega
        self.theta_bar = self.base.theta_bar
        self.time_step = self.base.time_step
        self.confidence_t = None

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
