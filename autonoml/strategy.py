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
                 do_random: bool = False, do_hpo: bool = True,
                 in_max_hpo_concurrency: int = 2,
                 in_n_iterations: int = 4, in_n_partitions: int = 3,
                 in_frac_validation: float = 0.25):
        
        if in_search_space is None:
            self.search_space = SearchSpace()
        else:
            self.search_space = in_search_space

        self.do_random = do_random
        self.do_hpo = do_hpo
        self.max_hpo_concurrency = in_max_hpo_concurrency

        self.n_iterations = in_n_iterations
        self.n_partitions = in_n_partitions
        self.frac_validation = in_frac_validation

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
    # TODO: Re-enable preprocessors when they work in HPO.
    for pool in [pool_predictors]:#, pool_preprocessors]:
        for typename_component, tuple_component in pool.items():
            hpars = tuple_component[1]
            hpar_space = {"Include": CustomBool(do_all_components)}
            if hpars:
                hpar_space["Hpars"] = hpars
            config_space[typename_component] = hpar_space
            count_component += 1

    strategy = Strategy()

    dict_strategy = {"Strategy": {"Do Random": CustomBool(strategy.do_random), 
                                  "Do HPO": CustomBool(strategy.do_hpo),
                                  "Max HPO Concurrency": strategy.max_hpo_concurrency},
                     "Optimiser": {"BOHB": {"Note": ("Prior to HPO, the dataset is randomly split into "
                                                     "a training fraction and a validation fraction. "
                                                     "For i iterations and p partitions, BOHB seeks to "
                                                     "sample p^i models on 1/p^i of training data at the 1st iteration, "
                                                     "then propagate the best 1/p models to the next iteration."),
                                            "Iterations": strategy.n_iterations, "Partitions": strategy.n_partitions, 
                                            "Validation Fraction": strategy.frac_validation}},
                     "Search Space": config_space}

    with open(in_filepath, "w") as file:
        yaml.dump(dict_strategy, file, default_flow_style = False, sort_keys = False,
                  Dumper = CustomDumper)
    log.info("%s - Template strategy generated at: %s" 
             % (Timestamp(), os.path.abspath(in_filepath)))

def import_strategy(in_filepath: str):

    with open(in_filepath, "r") as file:
        specs = yaml.safe_load(file)

    strategy = Strategy(in_search_space = SearchSpace(specs["Search Space"]),
                        do_random = bool(CustomBool(specs["Strategy"]["Do Random"])),
                        do_hpo = bool(CustomBool(specs["Strategy"]["Do HPO"])),
                        in_max_hpo_concurrency = int(specs["Strategy"]["Max HPO Concurrency"]),
                        in_n_iterations = int(specs["Optimiser"]["BOHB"]["Iterations"]),
                        in_n_partitions = int(specs["Optimiser"]["BOHB"]["Partitions"]),
                        in_frac_validation = float(specs["Optimiser"]["BOHB"]["Validation Fraction"]))

    return strategy