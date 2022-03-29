import json
import jsonschema
from jsonschema import validate
import math

def log_space_convert(solver,limitnumber,paramset,json_param_file,exp=False):

    if not exp:

        paramNames = list(json_param_file.keys())

        params = json_param_file

        maxval_indices = []
        minval_indices = []

        to_delete = []

        for i in range(len(paramNames)):
            if params[paramNames[i]]['paramtype'] == 'categorical':            
                one_hot_addition = []
                if params[paramNames[i]]['valtype'] == 'int':
                    minVal = params[paramNames[i]]['minval']
                    maxVal = params[paramNames[i]]['maxval']
                    valuerange = maxVal - minVal + 1
                    values = [minVal+j for j in range(valuerange)]
                    if len(values) > 2:
                        to_delete.append(i)

        for index in sorted(to_delete, reverse=True):
            del paramNames[index]


        for i in range(len(paramNames)):
            if params[paramNames[i]]['paramtype'] == 'continuous':
                if params[paramNames[i]]['maxval'] >= limitnumber:
                    maxval_indices.append(i)
                if params[paramNames[i]]['minval'] <= -limitnumber:
                    minval_indices.append(i)
            if params[paramNames[i]]['paramtype'] == 'discrete':
                if params[paramNames[i]]['maxval'] >= limitnumber:
                    maxval_indices.append(i)
                if params[paramNames[i]]['minval'] <= -limitnumber:
                    minval_indices.append(i)

        for i in range(len(minval_indices)):
            if minval_indices[i] in maxval_indices:
                del minval_indices[i]


        for i in range(len(paramset)):
            for j in range(len(maxval_indices)):
                if float(paramset[i][maxval_indices[j]]) > 0:
                    paramset[i][maxval_indices[j]] = \
                    math.log(float(paramset[i][maxval_indices[j]]))
                elif float(paramset[i][maxval_indices[j]]) < 0:
                    paramset[i][maxval_indices[j]] = \
                    -math.log(float(-paramset[i][maxval_indices[j]]))
            for j in range(len(minval_indices)):
                if float(paramset[i][minval_indices[j]]) < 0:
                    paramset[i][minval_indices[j]] = \
                    -math.log(float(-paramset[i][minval_indices[j]]))
                elif float(paramset[i][minval_indices[j]]) > 0:
                    paramset[i][minval_indices[j]] = \
                    math.log(float(paramset[i][minval_indices[j]]))


        return paramset

    if exp:

        paramNames = list(json_param_file.keys())

        params = json_param_file

        maxval_indices = []
        minval_indices = []

        to_delete = []

        for i in range(len(paramNames)):
            if params[paramNames[i]]['paramtype'] == 'categorical':            
                one_hot_addition = []
                if params[paramNames[i]]['valtype'] == 'int':
                    minVal = params[paramNames[i]]['minval']
                    maxVal = params[paramNames[i]]['maxval']
                    valuerange = maxVal - minVal + 1
                    values = [minVal+j for j in range(valuerange)]
                    if len(values) > 2:
                        to_delete.append(i)

        for i in range(len(paramNames)):
            if params[paramNames[i]]['paramtype'] == 'continuous':
                if params[paramNames[i]]['maxval'] >= limitnumber:
                    maxval_indices.append(i)
                if params[paramNames[i]]['minval'] <= -limitnumber:
                    minval_indices.append(i)
            if params[paramNames[i]]['paramtype'] == 'discrete':
                if params[paramNames[i]]['maxval'] >= limitnumber:
                    maxval_indices.append(i)
                if params[paramNames[i]]['minval'] <= -limitnumber:
                    minval_indices.append(i)

        for i in range(len(minval_indices)):
            if minval_indices[i] in maxval_indices:
                del minval_indices[i]

        for j in range(len(maxval_indices)):
            if float(paramset[maxval_indices[j]]) > 0:
                    paramset[maxval_indices[j]] = \
                    math.exp(float(paramset[maxval_indices[j]]))
            elif float(paramset[maxval_indices[j]]) < 0:
                    paramset[maxval_indices[j]] = \
                    -math.exp(float(-paramset[maxval_indices[j]]))
        for j in range(len(minval_indices)):
            if float(paramset[minval_indices[j]]) < 0:
                    paramset[minval_indices[j]] = \
                    -math.exp(float(-paramset[minval_indices[j]]))
            elif float(paramset[minval_indices[j]]) > 0:
                    paramset[minval_indices[j]] = \
                    math.exp(float(paramset[minval_indices[j]]))

        return paramset