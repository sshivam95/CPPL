import time
from itertools import count

from CPPL_class.CPPL import CPPLBase
import multiprocessing as mp

from Configuration_Functions import set_param


class Tournament:
    def __init__(self, cppl_base: CPPLBase, filename: str, contender_list) -> None:
        self.base = cppl_base
        self.subset_size = self.base.subset_size
        self.solver = self.base.args.solver
        self.timeout = self.base.args.timeout
        self.solver_parameter = self.base.solver_parameters
        self.pool = self.base.contender_pool
        self.baseline = self.base.args.baselineperf

        self.contender_list = contender_list

        self.filename = filename
        self.multiprocessing_event = mp.Event()
        mp.freeze_support()
        self.event = mp.Manager().list([0])
        self.winner = mp.Manager().list([None])
        self.results = mp.Manager().list([[0, 0] for _ in range(self.subset_size)])
        self.interim = mp.Manager().list([0 for _ in range(self.subset_size)])
        if self.solver == 'cadical':
            self.interim = mp.Manager().list([[0, 0, 0, 0, 0, 0] for _ in range(self.subset_size)])
        elif self.solver == 'glucose':
            self.interim = mp.Manager().list([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(self.subset_size)])
        elif self.solver == 'cplex':
            self.interim = mp.Manager().list([[1000000, 100, 0, 0] for _ in range(self.subset_size)])
        self.new_time = mp.Manager().list([self.timeout])
        self.process_ids = mp.Manager().list([[0] for _ in range(self.subset_size)])
        self.subset_start_time = mp.Manager().list([[0] for _ in range(self.subset_size)])

        # Initialize parallel solving data
        self.process = ["process_{0}".format(core) for core in range(self.subset_size)]
        self.parallel_results = [[0 for _ in range(2)] for _ in range(self.subset_size)]
        self.interim_result = [[0 for _ in range(3)] for _ in range(self.subset_size)]
        self.start_time = time.time()
        self.winner_known = True

    def get_process(self):
        for core in range(self.subset_size):
            contender = str(self.contender_list[core])

            parameter_str = set_param.set_contender_params(
                contender_index=contender,
                genes=self.pool[contender],
                solver_parameters=self.solver_parameter,
                return_it=True
            )

            self.process[core] = mp.Process(
                target=self.start_run,
                args=[
                    core
                ]# TODO add args
            )

        # Start processes
        for core in range(self.subset_size):
            self.process[core].start()  # start process of object Process()

        return self.process

    def start_run(self, core):
        self.subset_start_time[core] = time.process_time()

        pass


