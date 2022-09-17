"""A CPPL class which represents CPPL algorithm"""
import logging

from CPPL_class.cppl_configuration import CPPLConfiguration
from utils.utility_functions import (
    gradient,
)
from tournament_classes.tournament import Tournament


class CPPLAlgo(CPPLConfiguration):
    def __init__(
            self,
            args,
            logger_name="CPPLAlgo",
            logger_level=logging.INFO,
    ):
        super().__init__(args=args)
        self.tournament = None
        self.contender_list = None
        self.context_matrix = None
        self.current_contender_names = None
        self.current_pool = None
        self.solver = self.base.args.solver
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)
        self.winners_list = []

    def run(self):
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
                            discard,
                        ) = self._get_contender_list(filename=filename)

                        self.base.S_t = []  # S_t
                        for contender in self.contender_list:
                            self.base.S_t.append(
                                int(contender.replace("contender_", ""))
                            )

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

                    # Log Winners for instances
                    self.base.tracking_winners.info(self.winners_list)

                else:
                    # When directory has no more instances, break
                    self.base.is_finished = True

        print(
            "\n  #######################\n ",
            "Finished all instances!\n ",
            "#######################\n",
        )

    def update(self):
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

        self.base.theta_hat = (
                self.base.theta_hat
                + self.base.gamma
                * self.base.time_step ** (-self.base.alpha)
                * self.base.grad
        )
        self.base.theta_hat[self.base.theta_hat < 0] = 0
        self.base.theta_hat[self.base.theta_hat > 0] = 1

        # Update theta_bar
        self.base.theta_bar = (
                (self.base.time_step - 1) * self.base.theta_bar / self.base.time_step
                + self.base.theta_hat / self.base.time_step
        )
