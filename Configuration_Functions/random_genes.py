from random import randint
import random
import numpy as np
import math
import sys

import json
import jsonschema
from jsonschema import validate
from numpy.random import choice


def get_all_min_and_max(solver,json_param_file):

    params = json_param_file

    paramNames = list(params.keys())

    genes = [0 for i in range(len(paramNames))]

    all_min = []
    all_max = []

    for i in range(len(paramNames)):
        if params[paramNames[i]]['paramtype'] == 'categorical':
            if params[paramNames[i]]['valtype'] == 'str':
                values = params[paramNames[i]]['values']
                all_min.append(values[0])
                all_max.append(values[len(values)-1])

            if params[paramNames[i]]['valtype'] == 'int':
                all_max.append(params[paramNames[i]]['maxval'])
                all_min.append(params[paramNames[i]]['minval'])

        elif params[paramNames[i]]['paramtype'] == 'continuous':
            all_max.append(params[paramNames[i]]['maxval'])
            all_min.append(params[paramNames[i]]['minval'])

        elif params[paramNames[i]]['paramtype'] == 'discrete':
            all_max.append(params[paramNames[i]]['maxval'])
            all_min.append(params[paramNames[i]]['minval'])

        elif params[paramNames[i]]['paramtype'] == 'binary':
            all_max.append(1)
            all_min.append(0)

    return all_min, all_max

def genes_set(solver,json_param_file=None):

    if json_param_file == None:
        json_file_name = 'params_'+str(solver)

        with open(f'Configuration_Functions/{json_file_name}.json', 
                  'r') as f:
            data = f.read()
        params = json.loads(data)

        paramNames = list(params.keys())

    else:
        params = json_param_file
        paramNames = list(params.keys())

    genes = [0 for i in range(len(paramNames))]


    for i in range(len(paramNames)):
        if params[paramNames[i]]['paramtype'] == 'categorical':

            if params[paramNames[i]]['valtype'] == 'str':
                values = params[paramNames[i]]['values']

                if 'flag' in params[paramNames[i]]:
                    NameIsVal = True
                else:
                    Name = paramNames[i]
                    NameIsVal = False
                genes[i] = random.choice(values)
                if NameIsVal:
                    Name = genes[i]

            if params[paramNames[i]]['valtype'] == 'int':
                maxVal = params[paramNames[i]]['maxval']
                minVal = params[paramNames[i]]['minval']
                genes[i] = random.randint(minVal, maxVal)

        elif params[paramNames[i]]['paramtype'] == 'continuous':
            maxVal = params[paramNames[i]]['maxval']
            minVal = params[paramNames[i]]['minval']

            if 'splitbydefault' in params[paramNames[i]]:
                if params[paramNames[i]]['splitbydefault']:
                    default = params[paramNames[i]]['default']
                    splitbydefault = True
            else:
                splitbydefault = False

            if 'distribution' in params[paramNames[i]]:
                if params[paramNames[i]]['distribution'] == 'log':

                    logmaxVal=math.log(maxVal)
                    
                    weights = []
                    if minVal <= 0:
                        if 'includezero' in params[paramNames[i]]:
                            if params[paramNames[i]]['includezero'] == True:
                                includezero = True
                                if 'probabilityzero' in params[paramNames[i]]:
                                    if params[paramNames[i]]['probabilityzero'] == True:
                                        probabzero = params[paramNames[i]]['probabilityzero']
                                    else:
                                        # default probability if the probability 
                                        # for zero is not set in params_solver.json
                                        probabzero = 0.1
                                    weights.append(probabzero)
                        else:
                            includezero = False
                    else:
                        includezero = False

                    if 'logonpos' in params[paramNames[i]]:
                        if params[paramNames[i]]['logonpos'] == True:
                            logonpos = True
                        if 'probabpos' in params[paramNames[i]]:
                            probabpos = params[paramNames[i]]['probabpos']
                        else:
                            if probabzero and 'probabneg' in params[paramNames[i]]:
                                probabpos = 1 - probabzero - params[paramNames[i]]['probabneg']
                            elif probabzero:
                                probabpos = (1-probabzero)/2
                            else:
                                probabpos = 0.5
                        weights.append(probabpos)
                    else:
                        logonpos = False

                    if minVal < 0:
                        if 'logonneg' in params[paramNames[i]]:
                            if params[paramNames[i]]['logonneg'] == True:
                                logonneg = True
                                logminVal=math.log(-minVal)
                            if 'probabneg' in params[paramNames[i]]:
                                probabneg = params[paramNames[i]]['probabneg']
                            else:
                                if probabzero:
                                    probabneg = 1 - probabzero - probabpos
                                elif probabzero and probabpos == (1-probabzero)/2:
                                    probabneg = (1-probabzero)/2
                                else:
                                    probabneg = 0.5
                            weights = [probabneg] + weights
                        else:
                            logonneg = False
                    else:
                        logonneg = False
                                

                    if logonpos:
                        high = math.exp(np.random.uniform(0.000000001,logmaxVal,size=1))
                    else:
                        high = random.uniform(0.000000001,maxVal)
                    if splitbydefault:
                        if default == 0:
                            high = math.exp(np.random.uniform(default,logmaxVal,size=1))
                        else:
                            high = math.exp(np.random.uniform(math.log(default),logmaxVal,size=1))
                    if logonneg:
                        low = -math.exp(np.random.uniform(0.000000001,-logminVal,size=1))-minVal
                    else:
                        if splitbydefault:
                            low = random.uniform(minVal,default)
                        else:
                            low = random.uniform(minVal,0.000000001)

                    if len(weights) == 3:
                        genes[i] = float(choice([low,0,high],1,p=weights))
                    elif len(weights) == 2:
                        if not includezero:
                            genes[i] = float(choice([low,high],1,p=weights))
                        else:
                            weights[1] = 1 - probabzero
                            genes[i] = float(choice([0,high],1,p=weights))
                    if splitbydefault:
                        if logonpos:
                            weights = [1-probabpos,probabpos]
                        else:
                            weights = [0.5,0.5]
                        genes[i] = float(choice([low,high],1,p=weights))
                    if not logonneg and not logonpos and minVal > 0:
                        genes[i] = float(choice([math.exp(np.random.uniform(math.log(minVal),
                                         logmaxVal,size=1))],1)[0])

            else:
                genes[i] = random.uniform(minVal,maxVal)



        elif params[paramNames[i]]['paramtype'] == 'discrete':
            maxVal = params[paramNames[i]]['maxval']
            minVal = params[paramNames[i]]['minval']

            if 'splitbydefault' in params[paramNames[i]]:
                if params[paramNames[i]]['splitbydefault']:
                    default = params[paramNames[i]]['default']
                    splitbydefault = True
            else:
                splitbydefault = False

            if 'distribution' in params[paramNames[i]]:
                if params[paramNames[i]]['distribution'] == 'log':

                    logmaxVal=math.log(maxVal)
                    
                    weights = []
                    if minVal <= 0:
                        if 'includezero' in params[paramNames[i]]:
                            if params[paramNames[i]]['includezero'] == True:
                                includezero = True
                                if 'probabilityzero' in params[paramNames[i]]:
                                    if params[paramNames[i]]['probabilityzero'] == True:
                                        probabzero = params[paramNames[i]]['probabilityzero']
                                    else:
                                        # default probability if the probability
                                        # for zero is not set in params_solver.json
                                        probabzero = 0.1
                                    weights.append(probabzero)
                        else:
                            includezero = False
                    else:
                        includezero = False

                    if 'logonpos' in params[paramNames[i]]:
                        if params[paramNames[i]]['logonpos'] == True:
                            logonpos = True
                        if 'probabpos' in params[paramNames[i]]:
                            probabpos = params[paramNames[i]]['probabpos']
                        else:
                            if probabzero and 'probabneg' in params[paramNames[i]]:
                                probabpos = 1 - probabzero - params[paramNames[i]]['probabneg']
                            elif probabzero:
                                probabpos = (1-probabzero)/2
                            else:
                                probabpos = 0.5
                        weights.append(probabpos)
                    else:
                        logonpos = False

                    if minVal < 0:
                        if 'logonneg' in params[paramNames[i]]:
                            if params[paramNames[i]]['logonneg'] == True:
                                logonneg = True
                                logminVal=math.log(-minVal)
                            if 'probabneg' in params[paramNames[i]]:
                                probabneg = params[paramNames[i]]['probabneg']
                            else:
                                if probabzero:
                                    probabneg = 1 - probabzero - probabpos
                                elif probabzero and probabpos == (1-probabzero)/2:
                                    probabneg = (1-probabzero)/2
                                else:
                                    probabneg = 0.5
                            weights = [probabneg] + weights
                        else:
                            logonneg = False
                    else:
                        logonneg = False

                                

                    if logonpos:
                        high = math.exp(np.random.uniform(0.000000001,logmaxVal,size=1))
                    else:
                        high = random.uniform(0.000000001,maxVal)
                    if splitbydefault:
                        if default == 0:
                            high = math.exp(np.random.uniform(default,logmaxVal,size=1))
                        else:
                            high = math.exp(np.random.uniform(math.log(default),
                                                              logmaxVal,size=1))
                    if logonneg:
                        low = - math.exp(np.random.uniform(0.000000001,-logminVal,size=1)) \
                              - minVal
                    else:
                        if splitbydefault:
                            low = random.uniform(minVal,default)
                        else:
                            low = random.uniform(minVal,0.000000001)

                    if len(weights) == 3:
                        genes[i] = int(choice([low,0,high],1,p=weights))
                    elif len(weights) == 2:
                        if not includezero:
                            genes[i] = int(choice([low,high],1,p=weights))
                        else:
                            weights[1] = 1 - probabzero
                            genes[i] = int(choice([0,high],1,p=weights))
                    if splitbydefault:
                        if logonpos:
                            weights = [1-probabpos,probabpos]
                        else:
                            weights = [0.5,0.5]
                        genes[i] = int(choice([low,high],1,p=weights))

                    if not logonneg and not logonpos and minVal > 0:
                        genes[i] = int(choice([math.exp(np.random.uniform(math.log(minVal),
                                                        logmaxVal,size=1))],1))

            else:
                genes[i] = random.randint(minVal,maxVal)


        elif params[paramNames[i]]['paramtype'] == 'binary':
            default = params[paramNames[i]]['default']
            genes[i] = random.randint(0, 1)


    return genes#, params


def One_Hot_decode(genes,solver,param_value_dict=None,json_param_file=None,reverse=False):
    
    # Reverse One-Hot Encoding based on param_solver.json

    if json_param_file == None:
        json_file_name = 'params_'+str(solver)

        with open(f'Configuration_Functions/{json_file_name}.json', 'r') as f:
            data = f.read()
        params = json.loads(data)

        paramNames = list(params.keys())

    else:
        paramNames = list(json_param_file.keys())
        params = json_param_file

    genes = list(genes)


    if not reverse: # One-Hot decoding here (one-hot paramters back to solver specific representation)

        if len(genes) != len(paramNames):
            one_hot_tail = []
            insert_indices = []
            one_hot_vector_ranges = []
            real_vector = []

            for i in range(len(paramNames)):
                if params[paramNames[i]]['paramtype'] == 'categorical':
                    if params[paramNames[i]]['valtype'] == 'int':
                        minVal = params[paramNames[i]]['minval']
                        maxVal = params[paramNames[i]]['maxval']
                        if (maxVal - minVal +1) > 2:
                            one_hot_vector_ranges.append(maxVal-minVal+1)
                            real_vector.append([minVal+t for t in range(maxVal - minVal +1)])
                            one_hot_part = [0 for i in range(maxVal-minVal+1)]
                            one_hot_tail.append(one_hot_part)

            one_hot_tail_len = 0
            for i in range(len(one_hot_tail)):
                one_hot_tail_len += len(one_hot_tail[i])

            one_hot_values = genes[len(genes)-one_hot_tail_len:]

            cat_int_values = []

            for i in range(len(one_hot_tail)):
                cat_int_index = one_hot_values[:len(one_hot_tail[i])].index(1)
                cat_int_values.append(real_vector[i][cat_int_index])
                del one_hot_values[:len(one_hot_tail[i])]

            values_k = []
            for i in range(len(cat_int_values)):
                values_k.append(cat_int_values[i])

            new_genes = []
            insert_count = 0

            k = 0
            for key in param_value_dict:
                if param_value_dict[key] == None:
                    int_value = cat_int_values.pop(0)
                    new_genes.append(int_value)
                    insert_count += 1
                    k += 1
                else:
                    new_genes.append(genes[k-insert_count])
                    k += 1

            return new_genes

        else:
            return genes

    elif reverse: # one-hot encoding of previously one-hot decoded parameterizations
        Pool = genes
        P = []
        if len(genes[0]) == len(paramNames):
            decoded_count = 0
            genes = []
            param_value_dict = {}
            
            for j in range(len(Pool)):
                next_params = list(Pool[j])
                params = json_param_file
                originals_ind_to_delete = []
                one_hot_addition = []
                for i in range(len(paramNames)):
                    values = []
                    param_value_dict[paramNames[i]] = next_params[i]
                    if params[paramNames[i]]['paramtype'] == 'categorical':                                
                        if params[paramNames[i]]['valtype'] == 'int':
                            minVal = params[paramNames[i]]['minval']
                            maxVal = params[paramNames[i]]['maxval']
                            valuerange = int(maxVal) - int(minVal) + 1
                            values = [minVal+j for j in range(valuerange)]
                            index = int(next_params[i]) - minVal



                        if len(values) > 2:
                            # The categorical Parameter needs One-Hot Encoding
                            # -> append One-Hot Vectors and delete original elements                    
                            one_hot = [0 for j in range(len(values))]
                            one_hot[index] = 1
                            originals_ind_to_delete.append(i)
                            one_hot_addition += one_hot

                            param_value_dict[paramNames[i]] = None


                new_params = []

                for key in param_value_dict:
                    if param_value_dict[key] != None:
                        new_params.append(param_value_dict[key])

                new_params += one_hot_addition 

                P.append(new_params)
        else:
            P = Pool

        return P


def getParamsStringFromNumericParams(genes,solver,json_param_file=None):

    # Transform string categrials back to strings based on params_solver.json

    if json_param_file == None:
        json_file_name = 'params_'+str(solver)

        with open(f'Configuration_Functions/{json_file_name}.json', 'r') as f:
            data = f.read()
        params = json.loads(data)

        paramNames = list(params.keys())

    else:
        paramNames = list(json_param_file.keys())
        params = json_param_file

    genes = list(genes)

    for i in range(len(paramNames)):
        if params[paramNames[i]]['paramtype'] == 'categorical':
            if params[paramNames[i]]['valtype'] == 'str':
                values = params[paramNames[i]]['values']
                stringvalue = values[int(genes[i])]
                genes[i] = str(stringvalue)


    return genes