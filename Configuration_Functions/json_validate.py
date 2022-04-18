import json
import jsonschema
from jsonschema import validate


with open("params_cplex.json", "r") as f:
    data = f.read()
params = json.loads(data)

paramNames = list(params.keys())

with open("paramSchema.json", "r") as f:
    schema = f.read()
schemata = json.loads(schema)


def validateJson(jsonData):
    try:
        validate(instance=jsonData, schema=schemata)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True


for i in range(len(paramNames)):
    isValid = validateJson(params[paramNames[i]])
    if isValid:
        print("Yes")
    if not isValid:
        print(params[paramNames[i]])
        print("Given JSON data is InValid")

"""
# Convert json to python object.
jsonData = json.loads('{"name": "jane doe", "rollnumber": "25", "marks": 72}')
# validate it
isValid = validateJson(jsonData)
if isValid:
    print(jsonData)
    print("Given JSON data is Valid")
else:
    print(jsonData)
    print("Given JSON data is InValid")

# Convert json to python object.
jsonData = json.loads('{"name": "jane doe", "rollnumber": 25, "marks": 72}')
# validate it
isValid = validateJson(jsonData)
if isValid:
    print(jsonData)
    print("Given JSON data is Valid")
else:
    print(jsonData)
    print("Given JSON data is InValid")
"""
