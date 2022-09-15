"""Tournament file for running the tournament"""
import logging
import os
import signal
import time

from CPPL_class.CPPL import CPPLBase
import multiprocessing as mp

from utils import set_param
from tournament_classes import run_solver


class Tournament:
    """Tournament class for initialiting, runnning, watching and closing the parallel processes."""
    def __init__(
        self,
        cppl_base: CPPLBase,
        filepath: str,
        contender_list: list,
        logger_name="Tournament",
        logger_level=logging.INFO,
    ) -> None:
        """

        Parameters
        ----------
        cppl_base : CPPLBase
            Class object for the Base CPPL class.
        filepath : str
            Path of the problem instance to be solved by the solver.
        contender_list : list
            List of various contender or arms, each having different values of the parameters used by the
            solver to solve the problem instance.
        logger_name : str, default={Class name}
            Name of the logger.
        logger_level : int, default=logging.INFO
            Level of the logger.

        Attributes
        ----------
        base : CPPLBase
            Base class object.
        subset_size : int
            Size of subset of contender or arms. Equal to number of CPU processors in the system.
        solver : str
        """
        self.base = cppl_base
        self.subset_size = self.base.subset_size
        self.solver = self.base.args.solver
        self.timeout = self.base.args.timeout
        self.solver_parameter = self.base.solver_parameters
        self.pool = self.base.contender_pool
        self.baseline = self.base.args.baselineperf

        self.contender_list = contender_list
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.filename = filepath
        self.multiprocessing_event = mp.Event()
        mp.freeze_support()
        self.event = mp.Manager().list([0])
        self.winner = mp.Manager().list([None])
        self.process_results = mp.Manager().list(
            [[0, 0] for _ in range(self.subset_size)]
        )
        self.interim = mp.Manager().list([0 for _ in range(self.subset_size)])
        if self.solver == "cadical":
            self.interim = mp.Manager().list(
                [[0, 0, 0, 0, 0, 0] for _ in range(self.subset_size)]
            )
        elif self.solver == "glucose":
            self.interim = mp.Manager().list(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(self.subset_size)]
            )
        elif self.solver == "cplex":
            self.interim = mp.Manager().list(
                [[1000000, 100, 0, 0] for _ in range(self.subset_size)]
            )
        self.new_time = mp.Manager().list([self.timeout])
        self.process_ids = mp.Manager().list([[0] for _ in range(self.subset_size)])
        self.subset_start_time = mp.Manager().list(
            [[0] for _ in range(self.subset_size)]
        )

        # Initialize parallel solving data
        self.process = [f"process_{core}" for core in range(self.subset_size)]
        self.results = [[0 for _ in range(2)] for _ in range(self.subset_size)]
        self.interim_result = [[0 for _ in range(3)] for _ in range(self.subset_size)]
        self.start_time = time.time()

    def run(self) -> None:
        """

        Returns
        -------

        """
        for core in range(self.subset_size):
            contender = str(self.contender_list[core])

            parameter_str = set_param.set_contender_params(
                contender_index=contender,
                contender_pool=self.pool[contender],
                solver_parameters=self.solver_parameter,
                return_it=True,
            )

            self.process[core] = mp.Process(
                target=self.start_run, args=[core, parameter_str]
            )

        # Start processes
        for core in range(self.subset_size):
            self.process[core].start()  # start process of object Process()

    @staticmethod
    def non_nlock_read(output: str) -> str:
        # fd = output.fileno()
        # fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        # fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            return output.readline()
        except:
            return ""

    def start_run(self, core: int, parameter_str: list[str]) -> None:
        """

        Parameters
        ----------
        core :
        parameter_str :

        Returns
        -------

        """
        self.subset_start_time[core] = time.process_time()

        if self.baseline:
            params = []
        else:
            params = parameter_str

        process = run_solver.start(
            params=params,
            time_limit=self.timeout,
            filename=self.filename,
            solver=self.solver,
        )

        self.process_ids[core] = process.pid
        awaiting = True

        while awaiting:
            line = self.non_nlock_read(output=process.stdout)

            if line != b"":
                output = run_solver.check_output(
                    line=line, interim=self.interim[core], solver=self.solver
                )

                if output != "No output" and output is not None:
                    self.interim[core] = output

                is_solved = run_solver.check_if_solved(
                    line=line,
                    results=self.results[core],
                    proc=process,
                    event=self.event,
                    non_nlock_read=self.non_nlock_read,
                    solver=self.solver,
                )

                if is_solved != "No output":
                    self.results[core], self.event[0] = is_solved
                else:
                    print("########################")
                    print("Solver did not solve the instance")
                    print("########################")

            if self.results[core][0] != int(0):
                sub_now = time.process_time()
                self.results[core][1] = (
                    self.results[core][1] + sub_now - self.subset_start_time[core]
                )
                self.multiprocessing_event.set()
                self.event[0] = 1
                self.winner[0] = core
                self.process_results[core] = self.results[core][:]
                self.new_time[0] = self.results[core][1]
                awaiting = False

            if self.event[0] == 1 or self.multiprocessing_event.is_set():
                awaiting = False
                process.terminate()
                time.sleep(0.1)
                if process.poll() is None:  # if child process has terminated
                    process.kill()  # Kill the process
                    time.sleep(0.1)

                    for index in range(self.subset_size):
                        if (
                            self.subset_start_time[index] - time.process_time()
                            >= self.new_time[0]
                            and index != core
                        ):
                            os.kill(
                                __pid=self.process_ids[index], __signal=signal.SIGKILL
                            )

                    time.sleep(0.1)
                    try:
                        os.kill(__pid=process.pid, __signal=signal.SIGKILL)

                    except:
                        continue

                if self.solver == "cadical":
                    time.sleep(0.1)
                    for index in range(self.subset_size):
                        if (
                            self.subset_start_time[index] - time.process_time()
                            >= self.new_time[0]
                            and index != core
                        ):
                            try:
                                os.system("killall cadical")
                            except:
                                continue

    def watch_run(self) -> None:
        """

        Returns
        -------

        """
        while any(proc.is_alive() for proc in self.process):
            time.sleep(1)
            now = time.time()
            current_time = now - self.start_time

            if current_time >= self.timeout:
                self.multiprocessing_event.set()
                self.event[0] = 1
                for core in range(self.subset_size):
                    try:
                        os.kill(__pid=self.process_ids[core], __signal=signal.SIGKILL)
                    except:
                        continue

            if self.multiprocessing_event.is_set() or self.event[0] == 1:
                if self.solver == "cadical":
                    time.sleep(10)
                    if any(proc.is_alive() for proc in self.process):
                        try:
                            os.system("killall cadical")
                        except:
                            continue

    def close_run(self) -> None:
        """

        Returns
        -------

        """
        # Prepare interim for processing (str -> int)
        for core in range(self.subset_size):
            self.interim[core] = list(map(int, self.interim[core]))

        # Close processes
        for core in range(self.subset_size):
            self.process[core].join()

        for core in range(self.subset_size):
            self.results[core][:] = self.process_results[core][:]
            self.interim_result[core][:] = self.interim[core][:]
