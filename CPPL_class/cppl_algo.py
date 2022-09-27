"""An implementation of CPPL algorithm."""
from argparse import Namespace
import logging

from CPPL_class.cppl_configuration import CPPLConfiguration
from utils.utility_functions import (
    gradient,
)
from tournament_classes.tournament import Tournament


class CPPLAlgo(CPPLConfiguration):
    """A CPPL class which represents the working of the CPPL algorithm.

    Parameters
    ----------
    args : Namespace
        The arguments of the algorithm given by user.
    logger_name : str, optional
        Logger name, by default "CPPLAlgo"
    logger_level : int, optional
        Level of the logger, by default logging.INFO

    Attributes
    ----------
    tournament : Tournament
        An object of class Tournament.
    contender_list : List[str]
        List of contenders from the pool of contenders after preselection. If the winner is known with the highest upper bounds on the latent utility,
        the list contains the winner arm else the list is generated with arms having highest estimated latent utility. Note: this changes based on different
        preselection algorithms.
    context_matrix : np.ndarray
        A context matrix where each element is associated with one of the different arms and contains
        the properties of the arm itself as well as the context in which the arm needs to be chosen.
    current_contender_names : List[str]
        List of the contenders in the current time step.
    current_pool : List[int]
        The contenders pool after the update.
    solver : str
        Solver used to solve the instances.
    winners_list : List[int]
        A list of all winners over the time step.
    """

    def __init__(
        self,
        args: Namespace,
        logger_name: str = "CPPLAlgo",
        logger_level: int = logging.INFO,
    ) -> None:
        super().__init__(args=args)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.tournament: Tournament = None
        self.contender_list = None
        self.context_matrix = None
        self.current_contender_names = None
        self.current_pool = None
        self.solver = self.base.args.solver
        self.winners_list = []

    def run(self) -> None:
        """Run the algorithm until vompletion.

        Completion is determined by whether the solver has solved all the problem instance files.
        """
        # Read Instance file name to hand to solver
        # and check for format
        if self.solver == "cadical" or self.solver == "glucose":
            file_ending = ".cnf"
        else:
            file_ending = ".mps"
            
        while not self.base.is_finished:
            # Iterate through all Instances
            for filename in self.base.problem_instance_list:

                dot = filename.rfind(".")
                file_path = f"{self.base.directory}/" + str(filename)

                # Run parametrization on instances
                if (
                    filename[dot:] == file_ending
                ):  # Check if input file extension is same as required by solver
                    print(
                        "\n \n ######################## \n",
                        "STARTING A NEW INSTANCE!",
                        "\n ######################## \n \n",
                    )

                    if self.base.winner_known:
                        # Get contender list
                        # X_t: Context information
                        # Y_t: winner
                        # S_t: subset of contenders
                        (
                            self.context_matrix,
                            self.contender_list,
                            discard,  # The discarded contenders.
                        ) = self._get_contender_list(filename=filename)

                        # self.base.S_t = []
                        # for contender in self.contender_list:
                        #     self.base.S_t.append(
                        #         int(contender.replace("contender_", ""))
                        #     )  # The subset of contenders after preselection.

                        if discard:
                            self.base.time_step = 1
                        self.base.time_step += 1
                    else:
                        self.contender_list = self._contender_list_including_generated()

                    self.tournament = Tournament(
                        cppl_base=self.base,
                        filepath=file_path,
                        contender_list=self.contender_list,
                    )
                    self.tournament.run()

                    # Output Setting
                    if self.base.args.data == "y":
                        print("Prior contender data is used!\n")
                    print("Timeout set to", self.base.args.timeout, "seconds\n")
                    print(
                        "contender_pool size set to",
                        self.base.args.contenders,
                        "individuals\n",
                    )
                    if self.base.args.pws == "pws":
                        print("Custom individual injected\n")
                    else:
                        print("No custom Individual injected\n")
                    print(".\n.\n.\n.\n")

                    # Observe the run and stop it if one parameterization finished
                    self.tournament.watch_run()
                    self.tournament.close_run()

                    print(f"Instance {filename} was finished!\n")

                    # Update parameter set
                    if self.base.args.baselineperf:
                        self.tournament.winner[0] = None
                        self.base.winner_known = False

                    if self.tournament.winner[0] is not None:
                        self.update()
                    else:
                        self.base.winner_known = False

                    print(
                        f"Time needed: {round(self.tournament.new_best_time[0], 2)} seconds \n\n"
                    )

                    # Update solving times for instances
                    self.base.instance_execution_times.append(
                        round(self.tournament.new_best_time[0], 2)
                    )

                    # Log execution times
                    self.base.tracking_times.info(self.base.instance_execution_times)

                    # Log Winners of problem instances
                    self.base.tracking_winners.info(self.winners_list)
                else:
                    print(f"{filename} is not of correct extension for the solver.\n")
                    break

            # When directory has no more instances, break
            self.base.is_finished = True
            
        print(
            "\n  #######################\n ",
            "Finished all instances!\n ",
            "#######################\n",
        )
        print(f"Regret: {self.base.regret}")

    def update(self) -> None:
        """Update the contenders pool, winners list, gradient and mean estimated score parameters after solving a problem instance."""
        self.current_pool = []

        for keys in self.base.contender_pool:
            self.current_pool.append(self.base.contender_pool[keys])

        self.current_contender_names = []
        for index, _ in enumerate(self.contender_list):
            self.current_contender_names.append(
                str(self.base.contender_pool[self.contender_list[index]])
            )

        self.contender_list = []
        for i in range(self.base.subset_size):
            self.contender_list.append(f"contender_{str(self.base.S_t[i])}")
        self.base.Y_t = int(self.contender_list[self.tournament.winner[0]][10:])
        print(f"Winner is contender_{self.base.Y_t}")
        self.winners_list.append(self.base.Y_t)  # Track winners

        self.base.grad = gradient(
            theta=self.base.theta_hat,
            winner_arm=self.base.Y_t,
            subset_arms=self.base.S_t,
            context_matrix=self.context_matrix,
        )

        # Update theta_hat
        self.base.theta_hat = (
            self.base.theta_hat
            + self.base.gamma
            * self.base.time_step ** (-self.base.alpha)
            * self.base.grad
        )
        self.base.theta_hat[self.base.theta_hat < 0] = 0
        self.base.theta_hat[self.base.theta_hat > 0] = 1

        # Update theta_bar based on theta_hat and time step
        self.base.theta_bar = (
            (self.base.time_step - 1) * self.base.theta_bar / self.base.time_step
            + self.base.theta_hat / self.base.time_step
        )
