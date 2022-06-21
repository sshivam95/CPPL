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


def validate_json(json_data):
    try:
        validate(instance=json_data, schema=schemata)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True


for i in range(len(paramNames)):
    isValid = validate_json(params[paramNames[i]])
    if isValid:
        print("Yes")
    if not isValid:
        print(params[paramNames[i]])
        print("Given JSON data is InValid")

"""
# Convert json to python object.
json_data = json.loads('{"name": "jane doe", "rollnumber": "25", "marks": 72}')
# validate it
isValid = validate_json(json_data)
if isValid:
    print(json_data)
    print("Given JSON data is Valid")
else:
    print(json_data)
    print("Given JSON data is InValid")

# Convert json to python object.
json_data = json.loads('{"name": "jane doe", "rollnumber": 25, "marks": 72}')
# validate it
isValid = validate_json(json_data)
if isValid:
    print(json_data)
    print("Given JSON data is Valid")
else:
    print(json_data)
    print("Given JSON data is InValid")
"""
