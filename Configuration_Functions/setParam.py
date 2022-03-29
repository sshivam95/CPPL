import json
import jsonschema
from jsonschema import validate

def setParams(contender,genes,solver,json_param_file,return_it=False):


      paramNames = list(json_param_file.keys())
      params = json_param_file

      paramset = [0 for i in range(len(genes))]

      for i in range(len(genes)):
        if 'flag' in params[paramNames[i]]:
          paramset[i] = str(genes[i])
        else:
          if json_param_file[paramNames[i]]['paramtype'] == 'discrete':
            paramset[i] = str(paramNames[i]) + str(int(genes[i]))
          else:
            paramset[i] = str(paramNames[i]) + str(genes[i])

      with open('ParamPool/' + str(contender), 'w') as file:
        print(" ".join(paramset), file=file)

      if return_it:
        return paramset