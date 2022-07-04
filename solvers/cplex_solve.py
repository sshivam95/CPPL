# pylint: disable=import-error,too-few-public-methods,no-member
"""CPLEX solver."""
import argparse
from ast import literal_eval
import time

from docplex.mp.model_reader import ModelReader
from docplex.mp.progress import ProgressListener

# Read and Solve Instance with CPLEX
# Parse directory of instances, instance filename, max. time_step for solving
parser = argparse.ArgumentParser(description="solve this instance")
parser.add_argument(
    "-d", "--directory", type=str, default="Instances", help="Directory for instances"
)
parser.add_argument("-f", "--file", type=str, default="0000.txt", help="Instance file")
parser.add_argument(
    "-to",
    "--timeout",
    type=float,
    default=300,
    help="Stop solving the instance after x seconds",
)
parser.add_argument(
    "-g", "--genes", type=str, default="pws", help="CPLEX Parameter set"
)
parser.add_argument(
    "-gap", "--objgap", type=float, default=0.001, help="Gap at which to stop (float)"
)
args = parser.parse_args()

direct = args.directory
filename = args.file
time_out = args.timeout


def read_and_set_params(param_list):
    """
    Read and return set of parameters from the list.

    :param param_list:
    :return:
    """
    param_list = str(param_list.strip())
    param_list = list(param_list[1:-1].split(" "))
    param_list = list(filter(lambda a: a != "", param_list))

    for i, _ in enumerate(param_list):
        param_split = list(param_list[i].split("="))
        if len(param_split) == 2:
            param_name = param_split[0].strip("'")
            param_value = param_split[1].strip(",").strip("'")
            parameter_set = literal_eval("model." + param_name)
            parameter_set(literal_eval(param_value))


if __name__ == "__main__":
    # model = cplex.Cplex(direct+'/'+filename)
    mr = ModelReader()
    model = mr.read_model(direct + "/" + filename)

    start = time.process_time()

    model.parameters.timelimit.set(time_out)

    class Listener(ProgressListener):
        """Progress Listener."""

        def __init__(self, time_step, gap):
            """
            Class to keep track of time.

            :param time_step:
            :param gap:
            """
            ProgressListener.__init__(self)
            self._gap = gap
            self._time = time_step

        def notify_progress(self, data):
            """
            Notify the preogress.

            :param data:
            :return:
            """
            if data.has_incumbent:
                print("CI: %f" % data.current_objective)
                print("GI: %.2f" % (100.0 * data.mip_gap))
                print("BI:", data.best_bound)
                print("CNI:", data.current_nb_nodes)

                # If we are solving for longer than the specified time_step then
                # stop if we reach the predefined alternate MIP gap.
                if data.time > self._time and data.mip_gap < self._gap:
                    print("ABORTING")
                    self.abort()
            else:
                # print('No incumbent yet')
                pass

    listener = Listener(time_out, args.objgap)

    model.parameters.clocktype = 1
    model.parameters.timelimit = time_out
    model.add_progress_listener(Listener(time_out, args.objgap))

    read_and_set_params(args.genes)

    solution = model.solve()

    parameter = literal_eval("model.solve_details.time_step")

    obj = model.objective_value

    # End measuring CPU read and solve time_step
    end = time.process_time()
    cpu_time = end - start

    # tt = model.get_solve_details().time_step
    tt = model.solve_details.time
    result = ["result_{0}".format(s) for s in range(2)]
    result[0] = obj
    result[1] = tt

    print(result)
