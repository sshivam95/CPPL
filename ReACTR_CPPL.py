## Start ReACTR_CPPL by:
## python3 ReACTR_CPPL.py -d modular_kcnf... 
## ... -to 300 -p pws -s cadical

import time
import os
import os.path
import signal
import multiprocessing as mp
from multiprocessing import Queue, Pipe, Manager, Event
import argparse
import numpy as np
import random
from subprocess import Popen, PIPE, STDOUT, run
import subprocess
import sys
import fcntl
import math
from statistics import mean
import csv 
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pathlib
from scipy.linalg import logm, expm

import json
import jsonschema
from jsonschema import validate

from Configuration_Functions import CPPLConfig
from Configuration_Functions import setParam
from Configuration_Functions import pws
from Configuration_Functions import random_genes
from Configuration_Functions.random_genes import genes_set
from Configuration_Functions import tournament as solving
from Configuration_Functions import file_logging
from Configuration_Functions import log_on_huge_params


# Parse directory of instances, solver, max. time for single solving
parser = argparse.ArgumentParser(description='Start Tournaments')
parser.add_argument('-d', '--directory', type=str, 
                    default='No Problem Instance Directory given', 
	                help='Direcotry for instances')
parser.add_argument('-s', '--solver', type=str, default='No solver chosen', 
	                help='Solver/math. model as .py')
parser.add_argument('-to', '--timeout', type=int, default=300, 
	                help='''Stop solving single instance after 
	                (int) seconds [300]''')
parser.add_argument('-nc', '--contenders', type=int, default=30, 
	                help='The number of contenders [30]')
parser.add_argument('-keeptop', '--top_to_keep', type=int, default=2, 
	                help='''Number of top contenders to get chosen 
	                for game [2]''')
parser.add_argument('-p', '--pws', type=str, default=None, 
	                help='Custom Parameter Genome []')
parser.add_argument('-usedata', '--data', type=str, default=None, 
	                help='''Type y if prior gene and score data should 
	                be used []''')
parser.add_argument('-experimental', '--exp', type=str, default=None, 
                    help='''Type y if prior gene and score data should 
                    be experimental (Pool_exp.txt) []''')
parser.add_argument('-ch', '--chance', type=int, default=25, 
	                help='''Chance to replace gene randomly 
	                in percent (int: 0 - 100) [25]''')
parser.add_argument('-m', '--mutate', type=int, default=10, 
	                help='''Chance for mutation in crossover process 
	                in percent (int: 0 - 100) [10]''')
parser.add_argument('-k', '--kill', type=float, default=5, 
	                help='''Contenders with a variance higher than 
	                this are killed and replaced (float) [5]''')
parser.add_argument('-tn', '--train_number', type=float, default=None, 
                    help='''How many of the first instances are to 
                    be trained on before starting (int) [None] ''')
parser.add_argument('-tr', '--train_rounds', type=float, default=0, 
                    help='''How many rounds are the first -tn instances 
                    to be trained on (int) [1] ''')
parser.add_argument('-fo', '--file_order', type=str, default='ascending', 
                    help='''Specify the order by whitch the problem 
                    instances are solved''')
parser.add_argument('-nc_pca_f', '--nc_pca_f', type=int, default=7, 
                    help='''Number of the dimensions for the PCA of the 
                    instance features ''')
parser.add_argument('-nc_pca_p', '--nc_pca_p', type=int, default=10, 
                    help='''Number of the dimensions for the PCA of the 
                    parameter (features) ''')
parser.add_argument('-jfm', '--jfm', type=str, default='polynomial', 
                    help='''Mode of the joned feature map''')
parser.add_argument('-omega', '--omega', type=float, default=10, 
                    help='''Omega parameter for CPPL''')
parser.add_argument('-gamma', '--gamma', type=float, default=11, 
                    help='''Gamma parameter for CPPL''')
parser.add_argument('-alpha', '--alpha', type=float, default=0.1, 
                    help='''Alpha parameter for CPPL''')
parser.add_argument('-tfn', '--times_file_name', type=str, 
                    default='Times_per_instance_CPPL.txt', 
                    help='''Name of the file which the times needed 
                    to solve instances are tracked in''')
parser.add_argument('-pl', '--paramlimit', type=float, default=100000, 
                    help='''Limit for the possible absolute value of 
                    a parameter for it to be normed to log space 
                    before CPPL comptation''')
parser.add_argument('-bp', '--baselineperf', type=bool, default=False, 
                    help='''Set to true if only default 
                    parameterizations should run''')
args, unknown = parser.parse_known_args()


solver = args.solver


# Initialize output of times needed for solving one instance
if args.directory != 'No Problem Instance Directory given':
    directory = os.fsencode(args.directory)
    path, dirs, files = next(os.walk(args.directory))
    file_count = len(files)
    times_insts = []
    if args.file_order == 'ascending':
        problem_instance_list = sorted(os.listdir(directory))
        clean_pil = ['' for i in range(len(problem_instance_list))]
        for ii in range(len(problem_instance_list)):
            clean_pil[ii] = str(os.fsdecode(problem_instance_list[ii]))
        problem_instance_list = clean_pil
    elif args.file_order == 'descending':
        problem_instance_list = sorted(os.listdir(directory),reverse=True)
        clean_pil = ['' for i in range(len(problem_instance_list))]
        for ii in range(len(problem_instance_list)):
            clean_pil[ii] = str(os.fsdecode(problem_instance_list[ii]))
        problem_instance_list = clean_pil
    else:
        file_order = str(args.file_order)
        with open(f'{file_order}.txt', 'r') as file:
            problem_instance_list = eval(file.read())
    with open('problem_instance_list.txt', 'w') as file:
        print(problem_instance_list, file=file)
else:
    print('\n\nYou need to specify a directory containing the problem ' \
          'instances!\n\n**[-d directory_name]**\n\n')
    sys.exit(0)

if args.solver == 'No solver chosen':
    print('\nYou need to choose a solver!!!\n\n**[-s <solver_name>]**\n\n')
    sys.exit(0)

tracking_times = file_logging.tracking_files(args.times_file_name, 
                                            'ReACTR_CPPL', 'INFO')
tracking_Pool = file_logging.tracking_files('Pool.txt', 'ReACTR_Pool', 
                                            'INFO')

original_chance = args.chance
original_mutation = args.mutate


# Count available cores
n = mp.cpu_count()
original_timeout = args.timeout

Paramdir = pathlib.Path("ParamPool")
if Paramdir.exists():
    pass
else:
    os.mkdir("ParamPool")


# Ordering the results with this
def find_best(list1,number):
    list_new = sorted(list1.items(), key=lambda kv: kv[1][1])
    return list_new[:number]

def validate_param_json(solver):
        json_file_name = 'params_'+str(solver)

        with open(f'Configuration_Functions/{json_file_name}.json', 'r') as f:
            data = f.read()
        params = json.loads(data)

        paramNames = list(params.keys())

        with open('Configuration_Functions/paramSchema.json', 'r') as f:
            schema = f.read()
        schemata = json.loads(schema)


        def JsonValidation(jsonfile):
            try:
                validate(instance=jsonfile, schema=schemata)
            except jsonschema.exceptions.ValidationError as err:
                return False
            return True

        for pn in paramNames:
            valid = JsonValidation(params[pn])
            if not valid:
                print(params[pn])
                print("Invalid JSON data structure. Exiting.")
                sys.exit(0)


        return params

json_param_file = validate_param_json(solver)

# Initialize Pool
if args.data == None:
    pool_keys = ['contender_{0}'.format(c) 
                 for c in range(args.contenders)]
    Pool = dict.fromkeys(pool_keys, 0)
    if args.baselineperf:
        print('Baseline Performance Run (only default parameters)')
        for key in Pool:
            Pool[key] = pws.set_genes(solver,json_param_file)
            setParam.setParams(key,Pool[key],solver,json_param_file)
    else:
        for key in Pool:
            Pool[key] = genes_set(solver)           
            setParam.setParams(key,Pool[key],solver,
                               json_param_file)
        if args.pws != None:
            Pool['contender_0'] = pws.set_genes(solver,json_param_file)            
            setParam.setParams('contender_0',Pool['contender_0'],
                               solver,json_param_file)

elif args.data == 'y':
    if args.exp == None:
        Pool_file = 'Pool.txt'
    elif args.exp == 'y':
        Pool_file = f'Pool_exp_{solver}.txt'
    with open(f'{Pool_file}','r') as file:
        Pool = eval(file.read())
        for key in Pool:
            setParam.setParams(key,Pool[key],solver,json_param_file)

# Write Pool to textfile for solver to access parameter settings
tracking_Pool.info(Pool)

# If training is required, prepare
if args.train_number is not None:
    for r, d, f in sorted(os.walk(directory)):
        continue

rounds_to_train = int(args.train_rounds)

###################################################################

InstFeatdir = pathlib.Path("Instance_Features")
if InstFeatdir.exists():
    pass
else:
    print('\nWARNING!\n\nA directory named <Instance_Features> with a ' \
          '.csv file containing the instance features is necessary!')
    if args.directory != 'No Problem Instance Directory given':
        print('\nIt must be named: Features_'+str(directory)[2:-1]+'.csv')
    else:
                print('\nIt must be named: ' \
                      'Features_<problem_instance_directory_name>.csv')
    print('\ninstance1 feature_value1 feature_value2 ' \
          '....\ninstance2 feature_value1...')
    print('\nExiting...')
    sys.exit(0)

# read features
if os.path.isfile('Instance_Features/training_features_' +
                  str(directory)[2:-1]+'.csv'):
    pass
else:
    print('\n\nThere needs to be a file with training instance features ' \
          'named << training_features_'+str(directory)[2:-1]+'.csv >> in' \
          ' the directory Instance_Features\n\n')
    sys.exit(0)

features = []
train_list = []
directory = str(directory)[2:-1]
with open(f'Instance_Features/training_features_{directory}.csv', 
          'r') as csvFile:
    reader = csv.reader(csvFile)
    next(reader)
    for row in reader:
        if len(row[1]) != 0:
            next_features = row
            train_list.append(row[0])
            next_features.pop(0)
            next_features = [float(j) for j in next_features]
            features.append(next_features)
csvFile.close()

features = np.asarray(features)

standard_scaler = preprocessing.StandardScaler()

features = standard_scaler.fit_transform(features)

# PCA on features
no_comp_pca_features = args.nc_pca_f
pca_obj_inst = PCA(n_components=no_comp_pca_features)
pca_obj_inst.fit(features)

# Get parameters and apply PCA
params, param_value_dict = CPPLConfig.read_parametrizations(Pool,solver)

#print(params)

params = np.asarray(params)


all_min, all_max = random_genes.get_all_min_and_max(solver,json_param_file)

all_min, _ = CPPLConfig.read_parametrizations(Pool,solver,contender=all_min)
all_max, _ = CPPLConfig.read_parametrizations(Pool,solver,contender=all_max)


params = np.append(params,[all_min],axis=0)

params = np.append(params,[all_max],axis=0)

params = log_on_huge_params.log_space_convert(solver,float(args.paramlimit),
                                              params,json_param_file)

min_max_scaler = preprocessing.MinMaxScaler()

params = min_max_scaler.fit_transform(params)

no_comp_pca_params = args.nc_pca_p
pca_obj_params = PCA(n_components=no_comp_pca_params)
pca_obj_params.fit(params)

# other parameters
jfm = args.jfm #'polynomial'
 
if jfm == 'concatenation':
    d = no_comp_pca_features + no_comp_pca_params
elif jfm == 'kronecker':
    d = no_comp_pca_features * no_comp_pca_params
elif jfm == 'polynomial':
    d = 4
    for i in range((no_comp_pca_features + no_comp_pca_params)-2):
        d = d + 3 + i
        
#theta_hat = np.random.rand(d)
theta_hat = np.zeros(d)
theta_bar = theta_hat

grad_op_sum = np.zeros((d,d))
hess_sum = np.zeros((d,d))
omega = args.omega
gamma_1 = args.gamma
alpha = args.alpha
t = 0
Y_t = 0
S_t = [] 
grad = np.zeros(d)
#########################################

def non_nlock_read(output):
    fd = output.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        return output.readline()
    except:
        return ""


def start_run(filename,timelimit,params,core):
    substart[core] = time.clock()
    if args.baselineperf:
        parames = []
    else:
        pass
    proc = solving.start(params,timelimit,filename,solver)
    pid = proc.pid
    pids[core] = pid
    awaiting = True
    while awaiting:

        line = non_nlock_read(proc.stdout)
        
        if line != b'':
            output = solving.check_output(line,interim[core],solver)
            if output != 'No output' and output is not None:
                interim[core] = output
                
            if_solved = solving.check_if_solved(line,results[core],proc,
                                                event,non_nlock_read,
                                                solver)
            if if_solved != 'No output':
                results[core], event[0] = if_solved
       

        if results[core][0] != int(0):
            subnow = time.clock()
            results[core][1] = results[core][1] + subnow - substart[core]
            ev.set()
            event[0] = 1
            winner[0] = core
            res[core] = results[core][:]
            newtime[0] = results[core][1]
            awaiting = False

            
        if event[0] == 1 or ev.is_set():                  
            awaiting = False
            proc.terminate()
            time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
                time.sleep(0.1)
                for ii in range(n):
                    if substart[ii] - time.clock() >= newtime[0] and \
                       ii != core:
                        os.kill(pids[ii], signal.SIGKILL)
                time.sleep(0.1)
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    continue
            if solver == 'cadical':
                time.sleep(0.1)
                for ii in range(n):
                    if substart[ii] - time.clock() >= newtime[0] and \
                       ii != core:
                        try:
                            os.system("killall cadical")
                        except:
                            continue


def initialize_data_structs():
    # Initialize pickled data
    ev = mp.Event()
    mp.freeze_support()
    event = Manager().list([0])
    winner = Manager().list([None])
    res = Manager().list([[0,0] for core in range(n)])
    if solver == 'cadical':
        interim = Manager().list([[0,0,0,0,0,0] for core in range(n)])
    elif solver == 'glucose':
        interim = Manager().list([[0,0,0,0,0,0,0,0,0,0] for core in range(n)])
    elif solver == 'cplex':
        interim = Manager().list([[1000000,100,0,0] for core in range(n)])
    newtime = Manager().list([args.timeout])
    pids = Manager().list([[0] for core in range(n)])
    substart = Manager().list([[0] for core in range(n)])

    # Initialize parallel solving data
    process = ['process_{0}'.format(s) for s in range(n)]
    global results
    results = [[0 for s in range(2)] for c in range(n)]
    interim_res = [[0 for s in range(3)] for c in range(n)]
    start = time.time()
    winner_known = True

    return ev, event, winner, res, interim, newtime, pids, substart, \
           process, results, interim_res, start, winner_known


def tournament(n,contender_list,start_run,filename,Pool):
    for core in range(n):
        contender = str(contender_list[core])

        param_string = setParam.setParams(contender,Pool[contender],solver,
                                   json_param_file,return_it=True)

        process[core] = mp.Process(target=start_run, 
                                   args=[filename,args.timeout,
                                   param_string,core])

    # Starting processes
    for core in range(n):
        process[core].start()

    return process


def watch_run(process,start,n,ev,pids):
    x = 0
    while any(proc.is_alive() for proc in process):
        time.sleep(1)
        x = x + 1
        if x == 4:
            x = 0           
        now = time.time()
        currenttime = now - start 
        if currenttime >= args.timeout:
            ev.set()
            event[0] == 1
            for core in range(n):
                try:
                    os.kill(pids[core], signal.SIGKILL)
                except:
                    continue
        if ev.is_set() or event[0] == 1:
            if solver == 'cadical':
                time.sleep(10)
                if any(proc.is_alive() for proc in process):
                    try:
                        os.system("killall cadical")
                    except:
                        continue


def close_run(n,interim,process,res,interim_res):
    # Prepare interim for processing (str -> int)
    for core in range(n):
        interim[core] = list(map(int, interim[core]))
    
    # Close processes
    for core in range(n):
        process[core].join()
 
    for core in range(n):
            results[core][:] = res[core][:]
            interim_res[core][:] = interim[core][:]

    return results, interim_res


def cppl_update(Pool,contender_list,winner,Y_t,theta_hat,theta_bar,S_t,X_t,
                gamma_1,t,alpha):
    current_pool = []

    for keys in Pool:
        current_pool.append(Pool[keys])

    current_contender_names = []
    for i in range(len(contender_list)):
        current_contender_names.append(str(Pool[contender_list[i]]))

    contender_list = []
    for i in range(n):
        contender_list.append('contender_'+str(S_t[i]))
        
    Y_t = int(contender_list[winner[0]][10:])
    print(f'Winner is contender_{Y_t}')
    [theta_hat, theta_bar, grad] = \
    CPPLConfig.update(Y_t, theta_hat, theta_bar, 
                      S_t, X_t, gamma_1, t, alpha)

    return current_pool, current_contender_names, \
           theta_hat, theta_bar, grad, Y_t

winner_known = True


if __name__ == '__main__':

    t = 0

    run = True

    while run:   
        # Iterate through all Instances
        for filename in problem_instance_list:

            # Read Instance file name to hand to solver 
            # and check for format
            if solver == 'cadical' or solver == 'glucose':
                file_ending = '.cnf'
            elif solver == 'cplex':
                file_ending = '.mps'
            dot = filename.find('.')

            file_path = f'{directory}/'+str(filename)

            # Run parametrizations on instances
            if filename[dot:] == file_ending and \
               file_path not in train_list:  

                print('\n \n ######################## \n',
                      'STARTING A NEW INSTANCE!', 
                	  '\n ######################## \n \n')


                if winner_known:                         
                    # Get contender list from CPPLConfig.py         
                    X_t, contender_list, discard \
                        = CPPLConfig.get_contenders(directory, filename, pca_obj_inst, 
                                                    pca_obj_params, jfm, theta_bar, t, 
                                                    n, Y_t, S_t, grad, hess_sum, grad_op_sum, 
                                                    omega, solver, Pool,tracking_Pool,
                                                    min_max_scaler, standard_scaler,
                                                    float(args.paramlimit),
                                                    param_value_dict,json_param_file,args.exp)

                    S_t = []
                    for i in range(len(contender_list)):
                        S_t.append(int(contender_list[i].replace('contender_','')))

                    if discard:
                        t = 1

                    t = t + 1

                else:
                    contender_list = CPPLConfig.contender_list_including_generated(Pool,solver,
                                                                                   float(args.paramlimit),
                                                                                   params,json_param_file,
                                                                                   theta_bar,jfm,min_max_scaler,
                                                                                   pca_obj_params,standard_scaler,
                                                                                   pca_obj_inst,directory,filename,n)


                # Start run
                ev, event, winner, res, interim, newtime, pids, substart, process, \
                results, interim_res, start, winner_known = initialize_data_structs()

                process = tournament(n,contender_list,start_run,file_path,Pool)

                # Output Setting
                if args.data == 'y':
                    print('Prior contender data is used!\n')
                print('Timeout set to', args.timeout, 'seconds\n')
                print('Poolsize set to', args.contenders, 'individuals\n')
                if args.pws == 'pws':
                    print('Custom individual injected\n')
                else:
                    print('No custom Individual injected\n')
                print('.\n.\n.\n.\n')

                # Observe the run and stop it if one parameterization finished
                watch_run(process,start,n,ev,pids)

                results, interim_res = close_run(n,interim,process,res,interim_res)

                print(f'Instance {filename} was finished!\n')


                # Update CPPLConfig.py
                if args.baselineperf:
                    winner[0] = None
                    winner_known = False
                if winner[0] != None:
                    current_pool, current_contender_names, \
                    theta_hat, theta_bar, grad, Y_t = \
                    cppl_update(Pool,contender_list,winner,
                                Y_t,theta_hat,theta_bar,S_t,
                                X_t,gamma_1,t,alpha)
                else:
                    winner_known = False
                
                print('Time needed:',round(newtime[0],2),'seconds\n\n\n')

                # Update solving times for instances
                times_insts.append(round(newtime[0],2))
            
                # Log times needed for instances to file
                tracking_times.info(times_insts)


            

            # Manage Training of args.train_number instances for
            # args.train_rounds of times
            if args.train_number is not None:
                files = sorted(f)
                if instance_file == files[int(args.train_number)-1]:
                    if rounds_to_train != 0:
                        print('Training Round', 
                              int(args.train_rounds - rounds_to_train + 1),
                              'Completed.\nRestart Run')
                        rounds_to_train = rounds_to_train - 1
                        inst = inst - int(args.train_number)
                        break
                    else:
                        run = False
            else:
                run = False

        else:
        	# When directory has no more instances, break 
            break

print('\n  #######################\n ',
      'Finished all instances!\n ',
      '#######################\n')
