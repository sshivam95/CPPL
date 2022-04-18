import sys
import time
from docplex.mp.model import Model as model
import docplex.mp.solution
from docplex.mp.progress import ProgressListener
from docplex.mp.model_reader import ModelReader
from docplex.mp.sdetails import SolveDetails as sd
import numpy as np
import os
import argparse
import time
import ast
import re

## Read and Solve Instance with CPLEX

# Parse directory of instances, instance filename, max. time for solving
parser = argparse.ArgumentParser(description='solve this instance')
parser.add_argument('-d', '--directory', type=str,
                    default='Instances',
                    help='Direcotry for instances')
parser.add_argument('-f', '--file', type=str,
                    default='0000.txt',
                    help='Instance file')
parser.add_argument('-to', '--timeout', type=float,
                    default=300,
                    help='Stop solving the instance after x seconds')
parser.add_argument('-g', '--genes', type=str,
                    default='pws',
                    help='CPLEX Parameter set')
parser.add_argument('-gap', '--objgap', type=float,
                    default=0.001,
                    help='Gap at wich to stop (float)')
args = parser.parse_args()

direct = args.directory
filename = args.file
time_out = args.timeout


def read_and_set_params(model, paramlist):
    paramlist = str(paramlist.strip())
    paramlist = list(paramlist[1:-1].split(" "))
    paramlist = list(filter(lambda a: a != '', paramlist))

    for i in range(len(paramlist)):
        paramsplit = list(paramlist[i].split("="))
        if len(paramsplit) == 2:
            paramName = paramsplit[0].strip('\'')
            paramvalue = paramsplit[1].strip(',').strip('\'')
            parameter = eval('model.' + paramName)
            parameter(eval(paramvalue))


if __name__ == "__main__":
    # model = cplex.Cplex(direct+'/'+filename)
    mr = ModelReader()
    model = mr.read_model(direct + '/' + filename)

    start = time.clock()

    model.parameters.timelimit.set(time_out)


    class Listener(ProgressListener):
        def __init__(self, time, gap):
            ProgressListener.__init__(self)
            self._gap = gap
            self._time = time

        def notify_progress(self, data):
            if data.has_incumbent:
                print('CI: %f' % data.current_objective)
                print('GI: %.2f' % (100. * data.mip_gap))
                print('BI:', data.best_bound)
                print('CNI:', data.current_nb_nodes)

                # If we are solving for longer than the specified time then
                # stop if we reach the predefined alternate MIP gap.
                if data.time > self._time and data.mip_gap < self._gap:
                    print('ABORTING')
                    self.abort()
            else:
                # print('No incumbent yet')
                pass


    listener = Listener(time_out, args.objgap)

    model.parameters.clocktype = 1
    model.parameters.timelimit = time_out
    model.add_progress_listener(Listener(time_out, args.objgap))

    read_and_set_params(model, args.genes)

    solution = model.solve()

    parameter = eval('model.solve_details.time')

    obj = model.objective_value

    # End measuring CPU read and solve time
    end = time.clock()
    cpu_time = end - start

    # tt = model.get_solve_details().time
    tt = model.solve_details.time
    result = ['result_{0}'.format(s) for s in range(2)]
    result[0] = obj
    result[1] = tt

    print(result)


    def mathmodel():
        return m


    def getobj():
        print('CURRENT VALUE IN CPLEXSOLVE', model.solution.get_objective_value())
        return model.solution.get_objective_value()
