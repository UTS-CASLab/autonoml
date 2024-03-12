import autonoml as aml

from copy import deepcopy

strategy = aml.import_strategy("./template.strat")

pipelines = aml.hpo.create_pipelines_default(in_keys_features = ["a"], in_key_target = "b", in_strategy = strategy)
model = pipelines[0].components[0].model

model_copy = deepcopy(model)