# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:30:58 2023

@author: David J. Kedziora
"""

from . import components
from .hyperparameter import HPInt, HPFloat
from .component import MLComponent, MLPredictor, MLPreprocessor
from .pipeline import MLPipeline
from .utils import log, Timestamp, CustomBool

import pkgutil
import importlib
import os

import yaml
# from ruamel.yaml import YAML

import numpy as np

pool_preprocessors = dict()
pool_predictors = dict()

# Import all modules within the components folder.
for importer, module_name, is_pkg in pkgutil.iter_modules(components.__path__):
    full_module_name = "%s.%s" % (components.__name__, module_name)
    loaded_module = importlib.import_module(full_module_name)
        
    # Find all classes in the loaded modules that are MLComponents.
    for name, obj in vars(loaded_module).items():
        if isinstance(obj, type) and issubclass(obj, MLComponent):

            # Consider ones that are as deep in their hierarchy without becoming unselectable.
            # Note: The False in getattr is the value if the attribute cannot be found.
            if not getattr(obj, "is_unselectable", False):
                subclasses = obj.__subclasses__()
                if (len(subclasses) == 0 
                    or all(getattr(subclass, "is_unselectable", False) for subclass in subclasses)):
                    
                    if issubclass(obj, MLPreprocessor):
                        pool_preprocessors[obj.__name__] = (obj, obj.new_hpars())
                    elif issubclass(obj, MLPredictor):
                        pool_predictors[obj.__name__] = (obj, obj.new_hpars())

class SearchSpace(dict):

    def list_predictors(self):
        categories = list()
        for typename_component in self:
            do_include = CustomBool(self[typename_component]["Include"])
            if do_include:
                if typename_component in pool_predictors:
                    categories.append(typename_component)
        return categories

class Strategy:

    def __init__(self, in_search_space: SearchSpace = None,
                 do_hpo: bool = False, do_custom: bool = False):
        
        self.search_space = SearchSpace()
        if not in_search_space is None:
            self.search_space = in_search_space

        self.do_hpo = do_hpo
        self.do_custom = do_custom

# print(pool_preprocessors)
# print(pool_predictors)

# def hyperparameter_constructor(loader, node):
#     values = loader.construct_mapping(node)
#     return Hyperparameter(**values)

# # Define a custom representer for the Hyperparameter class.
# def hyperparameter_representer(dumper, data):
#     return dumper.represent_dict(data.to_dict())

# class YAMLConstructor(ruamel.yaml.SafeConstructor): pass
# class YAMLRepresenter(ruamel.yaml.SafeRepresenter): pass

# YAMLConstructor.add_constructor("!Hyperparameter", hyperparameter_constructor)
# YAMLRepresenter.add_representer(Hyperparameter, hyperparameter_representer)
# yaml = ruamel.yaml.YAML()
# yaml.default_flow_style = False
# # yaml.sort_keys = False
# # yaml.Constructor = YAMLConstructor
# yaml.Representer = YAMLRepresenter

# def template_strategy(in_filepath: str = "./template.strat"):
#     global yaml

#     strategy = {component_name: tuple_component[1] 
#                 for component_name, tuple_component in pool_predictors.items()}
    
#     # strategy = OrderedDict([(component_name, tuple_component[1]) 
#     #                         for component_name, tuple_component in pool_predictors.items()])

#     with open(in_filepath, "w") as file:
#         yaml.dump(strategy, file)
#     log.info("%s - Template strategy generated at: %s" 
#              % (Timestamp(), os.path.abspath(in_filepath)))

#     # TODO: No returns.
#     return strategy

# def import_strategy(in_filepath: str):
#     global yaml

#     with open(in_filepath, "r") as file:
#         strategy = yaml.safe_load(file)

#     return strategy


# def hyperparameter_constructor(loader, node):
#     values = loader.construct_mapping(node)
#     return Hyperparameter(**values)

# # Define a custom representer for the Hyperparameter class.
# def hyperparameter_representer(dumper, data):
#     return dumper.represent_dict(data.to_dict_config())

# yaml.add_representer(Hyperparameter, hyperparameter_representer)

# yaml = YAML()
# yaml.representer.add_representer(Hyperparameter, hyperparameter_representer)

class CustomDumper(yaml.Dumper): pass
def custom_bool_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", repr(data))
CustomDumper.add_representer(CustomBool, custom_bool_representer)

def template_strategy(in_filepath: str = "./template.strat", 
                      do_all_components = True, do_all_hyperparameters = True):

    # Create/overwrite the YAML dumper Hyperparameter representer based on user requirements.
    def hyperparameter_representer(dumper, data):
        return dumper.represent_dict(data.to_dict_config(do_vary = do_all_hyperparameters))
    CustomDumper.add_representer(HPInt, hyperparameter_representer)
    CustomDumper.add_representer(HPFloat, hyperparameter_representer)

    config_space = dict()
    count_component = 0
    for pool in [pool_preprocessors, pool_predictors]:
        for typename_component, tuple_component in pool.items():
            hpars = tuple_component[1]
            hpar_space = {"Include": CustomBool(do_all_components)}
            if hpars:
                hpar_space["Hpars"] = hpars
            # config_space["&c" + str(count_component) + " " + typename_component] = hpar_space
            config_space[typename_component] = hpar_space
            count_component += 1

    strategy = {"Strategy": 
                {"Do HPO": CustomBool(True),
                 "Do Custom Pipelines": CustomBool(False)},
                "Search Space": config_space,
                "Custom Pipelines": ["a", "b"]}

    with open(in_filepath, "w") as file:
        yaml.dump(strategy, file, default_flow_style = False, sort_keys = False,
                  Dumper = CustomDumper)
    log.info("%s - Template strategy generated at: %s" 
             % (Timestamp(), os.path.abspath(in_filepath)))

def import_strategy(in_filepath: str):

    with open(in_filepath, "r") as file:
        specs = yaml.safe_load(file)

    strategy = Strategy(in_search_space = SearchSpace(specs["Search Space"]),
                        do_hpo = bool(CustomBool(specs["Strategy"]["Do HPO"])),
                        do_custom = bool(CustomBool(specs["Strategy"]["Do Custom Pipelines"])))

    return strategy

# TODO: Make truly random.
def create_pipeline_random(in_keys_features, in_key_target):

    predictor_cls = np.random.choice(list(tuple[0] for tuple in pool_predictors.values()))
    predictor_cls = pool_predictors["OnlineLinearRegressor"][0]
    structure = [predictor_cls(do_random_hpars = True)]

    pipeline = MLPipeline(in_keys_features = in_keys_features, in_key_target = in_key_target, 
                          in_components = structure)
    
    return pipeline