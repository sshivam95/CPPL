"""Tournament file for running the tournament"""
import logging
import os
import signal
import time
from typing import List

from CPPL_class.cppl_base import CPPLBase
import multiprocessing as mp

from utils import set_param
from tournament_classes import run_solver


class Tournament:
    """Tournament class for initialiting, runnning, watching and closing the parallel processes.
    Parameters
    ----------
    cppl_base : CPPLBase
        Class object for the Base CPPL class.
    filepath : str
        Path of the problem instance to be solved by the solver.
    contender_list : List
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
        Solver used to solve the instances.
    timeout : int
        The maximum time the solver can take to solve the instance problem. Once timeout is reached, the process will be terminated.
    solver_parameter : dict
        The parameter set used by the solver.
    pool : dict
        The contender or arms pool.
    baseline: bool
        Set to true if only default parameterization of the solver needs to be run.
    multiprocessing_event : multiprocessing.Event
        An asyncio event object can be used to notify multiple asyncio tasks that some event has happened.
    event : List[int]
        A shared list object of the events in the tournament.
    winner : List[int]
        A shared list object of the winner in the tournament. It contains only a single element which is the winner among the arms in the subset participating in the tournament.
    process_results : List[int]
        A shared list object of the process's results in the tournament.
    interim : List[int]
        A shared list object of the interim outputs of a process based on the solver in the tournament.
    new_best_time : List[int]
        A shared list object of the new best time of the winner after each round of the tournament.
    process_ids : List[int]
        A shared list object of the process ids in the tournament.
    substart_start_time : List[int]
        A shared list object of the start time of each arm in the subset.
    process : List[str]
        A list of process names in the tournament.
    interim_results : List[int]
        A list of all the interim results of the solvers in the tournament.
    start_time : float
        The start time of the tournament.
    """

    def __init__(
        self,
        cppl_base: CPPLBase,
        filepath: str,
        contender_list: List,
        logger_name: str = "Tournament",
        logger_level: int = logging.INFO,
    ) -> None:

        self.base = cppl_base
        self.filename = filepath
        self.contender_list = contender_list
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logger_level)

        self.subset_size = self.base.subset_size
        self.solver = self.base.args.solver
        self.timeout = self.base.args.timeout
        self.solver_parameter = self.base.solver_parameters
        self.pool = self.base.contender_pool
        self.baseline = self.base.args.baselineperf

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
        self.new_best_time = mp.Manager().list([self.timeout])
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
        """Run the tournament on a subset of contenders from the contender pool."""
        for core in range(self.subset_size):
            contender = str(self.contender_list[core])

            contender_parameter_set = set_param.set_contender_params(
                contender_index=contender,
                contender_pool=self.pool[contender],
                solver_parameters=self.solver_parameter,
                return_it=True,
            )

            self.process[core] = mp.Process(
                target=self.start_run, args=[core, contender_parameter_set]
            )

        # Start processes
        for core in range(self.subset_size):
            self.process[core].start()  # start process of object Process()

    @staticmethod
    def non_nlock_read(output: str) -> str:
        """Read the output from the solver on the problem instance.

        Parameters
        ----------
        output : str
            Output from the solver for the problem instance.

        Returns
        -------
        str
            Return the single line from the output parameter.
        """
        # fd = output.fileno()
        # fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        # fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            return output.readline()
        except:
            return ""

    def start_run(self, core: int, contender_parameter_set: List[str]) -> None:
        """Start running each process in the tournament.

        Each process represents each arm in the subset from the pool.

        Parameters
        ----------
        core : int
            The index of the arm. Here, the index represents the arm number as well.
        contender_parameter_set : List[str]
            The parameters of the arm related to the index.
        """
        self.subset_start_time[core] = time.process_time()

        if self.baseline:
            params = []
        else:
            params = contender_parameter_set

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
                self.new_best_time[0] = self.results[core][1]
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
                            >= self.new_best_time[0]
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
                            >= self.new_best_time[0]
                            and index != core
                        ):
                            try:
                                os.system("killall cadical")
                            except:
                                continue

    def watch_run(self) -> None:
        """Check if any process has terminated."""
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
        """If the execution of a process has stopped, update the results of the run."""
        # Prepare interim for processing (str -> int)
        for core in range(self.subset_size):
            self.interim[core] = list(map(int, self.interim[core]))

        # Close processes
        for core in range(self.subset_size):
            self.process[core].join()

        for core in range(self.subset_size):
            self.results[core][:] = self.process_results[core][:]
            self.interim_result[core][:] = self.interim[core][:]
