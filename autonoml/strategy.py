# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:30:58 2023

@author: David J. Kedziora
"""

from . import components
from .hyperparameter import HPInt, HPFloat
from .component import MLComponent, MLPredictor, MLPreprocessor
from .pipeline import MLPipeline
from .utils import log, Timestamp, CustomBool, flatten_dict

import pkgutil
import importlib
import os

import yaml

# pool_preprocessors = dict()
# pool_predictors = dict()

class ComponentCatalogue():
    """
    Provides a complete dictionary of all the components available in the package.
    """
    def __init__(self):
        # Set up a mapping from a module/name tuple to the actual component types.
        self.components = dict()
        
        # Set up a mapping between IDs of components and the modules they came from. 
        # self.cid_to_module = dict()
        self.module_to_cids = dict()

        # Set up a mapping between IDs of components and their categories.
        # For example, this will identify if a component is a preprocessor.
        self.categories = [MLPreprocessor, MLPredictor]
        self.cid_to_categories = dict()
        self.category_to_cids = {category: dict() for category in self.categories}

        self.compile()

    def compile(self):
        """
        Go through the components folder and construct a catalogue of available components.
        """
        # id_component = 0

        # Import all modules within the components folder.
        for _, module_short, _ in pkgutil.iter_modules(components.__path__):
            module_full = "%s.%s" % (components.__name__, module_short)
            loaded_module = importlib.import_module(module_full)

            self.module_to_cids[module_full] = dict()
                
            # Find all classes in the loaded modules that are MLComponents.
            for _, obj in vars(loaded_module).items():
                if isinstance(obj, type) and issubclass(obj, MLComponent):

                    # Consider ones that are as deep in their hierarchy without becoming unselectable.
                    # Note: The False in getattr is the value if the attribute cannot be found.
                    if not getattr(obj, "is_unselectable", False):
                        subclasses = obj.__subclasses__()
                        if (len(subclasses) == 0 
                            or all(getattr(subclass, "is_unselectable", False) for subclass in subclasses)):

                            id_component = ".".join([obj.__module__, obj.__name__])

                            self.components[id_component] = obj
                            # self.cid_to_module[id_component] = module
                            self.module_to_cids[module_full][id_component] = True

                            self.cid_to_categories[id_component] = list()
                            for category in self.categories:
                                if issubclass(obj, category):
                                    self.cid_to_categories[id_component].append(category)
                                    self.category_to_cids[category][id_component] = True

catalogue = ComponentCatalogue()

class SearchSpace(dict):
    """
    A dictionary of components/hyperparameters representing user choices for an ML problem.
    """
    def list_predictors(self):
        categories = list()
        for id_component in self:
            do_include = CustomBool(self[id_component]["Include"])
            if do_include:
                if id_component in catalogue.category_to_cids[MLPredictor]:
                    categories.append(id_component)
        return categories

class Strategy:
    def __init__(self, in_search_space: SearchSpace = None,
                 in_n_challengers: int = 2,
                 do_defaults: bool = False, do_random: bool = False, do_hpo: bool = False,
                 in_n_samples: int = 10,
                 in_max_hpo_concurrency: int = 2,
                 in_n_iterations: int = 4, in_n_partitions: int = 3,
                 in_frac_validation: float = 0.25,
                 in_folds_validation: int = 1):
        
        if in_search_space is None:
            self.search_space = SearchSpace()
        else:
            self.search_space = in_search_space

        self.n_challengers = in_n_challengers

        self.do_defaults = do_defaults
        self.do_random = do_random
        self.n_samples = in_n_samples
        self.do_hpo = do_hpo
        self.max_hpo_concurrency = in_max_hpo_concurrency

        self.frac_validation = in_frac_validation
        self.folds_validation = in_folds_validation

        self.n_iterations = in_n_iterations
        self.n_partitions = in_n_partitions

class CustomDumper(yaml.Dumper): pass
def custom_bool_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", repr(data))
CustomDumper.add_representer(CustomBool, custom_bool_representer)

def template_strategy(in_filepath: str = "./template.strat", 
                      do_all_components: bool = False, do_all_hyperparameters: bool = True):
    """
    Generate a template file for the user to dictate how a problem will be solved.
    """
    # Create/overwrite the YAML dumper Hyperparameter representer based on user requirements.
    def hyperparameter_representer(dumper, data):
        return dumper.represent_dict(data.to_dict_config(do_vary = do_all_hyperparameters))
    CustomDumper.add_representer(HPInt, hyperparameter_representer)
    CustomDumper.add_representer(HPFloat, hyperparameter_representer)

    overview = ("This file dictates how an AutonoMachine will try "
                "to learn the solution to an ML problem. "
                "Edit it as desired and import it when starting a learning process.")
    
    info_structure = ("An AutonoML solution consists of learner groups "
                      "allocated to different partitions of data. "
                      "In the base case, there is one group and it learns/adapts on all observations. "
                      "Each learner group has one champion and a number of challengers. "
                      "Newly developed pipelines exceeding this number will either kick out "
                      "an existing pipeline or fail to enter the ranks.")
    
    info_development = ("The defaults strategy starts the learning process with one ML pipeline "
                        "per predictor that has been enabled in the search space below, "
                        "where all hyperparameters are set to default values. "
                        "The random strategy starts with a number of pipelines and hyperparameters "
                        "randomly sampled from the enabled search space. "
                        "The HPO strategy runs a thread/process intensive hyperparameter optimisation "
                        "to find one optimal pipeline from the enabled search space. "
                        "It is recommended not to run too many HPOs concurrently. "
                        "Note that these strategies are applied once per learner group, in the case "
                        "where problem-solving involves allocations of data-subsets.")
    
    info_validation = ("A pipeline is initially scored on a validation fraction of "
                       "training data, averaged over a number of folds.")
    
    info_adaptation = ("By default, all pipeline components will adapt to new observations.")

    info_bohb = ("Prior to HPO, the dataset is randomly split into "
                 "a training fraction and a validation fraction. "
                 "For i iterations and p partitions, BOHB seeks to "
                 "sample p^i models on 1/p^i of training data at the 1st iteration, "
                 "then propagate the best 1/p models to the next iteration.")
    
    info_override = ("These options are quick ways to include/exclude categories of components. "
                     "Mark them y/n if desired, but leave them unmarked otherwise. "
                     "Otherwise, they will override any specific choices in the search space. "
                     "Also decide whether inclusions or exclusions are higher priority, "
                     "in the case that a component exists in multiple categories.")
    
    # Go through the component catalogue and create dictionaries to export into the template file.
    # The dictionaries should list modules and components/hyperparameters for inclusion/exclusion.
    config_space = dict()
    module_overrides = dict()

    for id_component, component in catalogue.components.items():

        module, _ = id_component.rsplit('.', 1)
        if not module in module_overrides:
            module_overrides[module] = ""

        hpars = component.new_hpars()
        
        hpar_space = {"Info": ", ".join([category.__name__ for category 
                                         in catalogue.cid_to_categories[id_component]]),
                      "Include": CustomBool(do_all_components)}
        if hpars:
            hpar_space["Hpars"] = hpars
        config_space[id_component] = hpar_space

    # Also add categories to inclusions/exclusions.
    overrides = {"Info": info_override,
                 "Prioritise Inclusions": CustomBool(True),
                 "Modules": module_overrides,
                 "Categories": {category.__name__: "" for category in catalogue.categories}}

    # Export a file that represents a default strategy.
    strategy = Strategy()

    dict_strategy = {"Overview": overview,
                     "Strategy": {"Structure": {"Info": info_structure,
                                                "Number of Challengers": strategy.n_challengers},
                                  "Development": {"Info": info_development,
                                                  "Do Defaults": CustomBool(strategy.do_defaults),
                                                  "Do Random": CustomBool(strategy.do_random),
                                                  "Number of Samples": strategy.n_samples,
                                                  "Do HPO": CustomBool(strategy.do_hpo),
                                                  "Max HPO Concurrency": strategy.max_hpo_concurrency},
                                  "Validation": {"Info": info_validation,
                                                 "Validation Fraction": strategy.frac_validation,
                                                 "Validation Folds": strategy.folds_validation},
                                  "Adaptation": {"Info": info_adaptation}},
                     "Optimiser": {"BOHB": {"Info": info_bohb,
                                            "Iterations": strategy.n_iterations, 
                                            "Partitions": strategy.n_partitions}}, 
                     "Inclusions/Exclusions": overrides,
                     "Search Space": config_space}

    with open(in_filepath, "w") as file:
        yaml.dump(dict_strategy, file, default_flow_style = False, sort_keys = False,
                  Dumper = CustomDumper)
    log.info("%s - Template strategy generated at: %s" 
             % (Timestamp(), os.path.abspath(in_filepath)))

# TODO: Develop a migration process based on version to update old strategy files.
# TODO: Create constants for magic-value strings to ensure there are no inconsistencies.
def import_strategy(in_filepath: str):

    log.info("%s - Importing strategy from: %s" 
             % (Timestamp(), os.path.abspath(in_filepath)))

    with open(in_filepath, "r") as file:
        specs = yaml.safe_load(file)

    # Deal with user-specified search-space overrides.
    # Note: Whatever override is prioritised must be applied last.
    search_space = SearchSpace(specs["Search Space"])
    overrides = specs["Inclusions/Exclusions"]
    if CustomBool(overrides["Prioritise Inclusions"]):
        order = [False, True]
    else:
        order = [True, False]
    for override_flag in order:
        for module in overrides["Modules"]:
            # CustomBool does not accept blank values, so roll on with any exceptions.
            # If a module does not exist in this version of the package, roll with the exception too.
            try:
                user_choice = overrides["Modules"][module]
                if bool(CustomBool(user_choice)) == override_flag:
                    for id_component in catalogue.module_to_cids[module]:
                        if id_component in search_space:
                            search_space[id_component]["Include"] = user_choice
            except:
                continue
        for category in catalogue.categories:
            if category.__name__ in overrides["Categories"]:
                try:
                    user_choice = overrides["Categories"][category.__name__]
                    if bool(CustomBool(user_choice)) == override_flag:
                        for id_component in catalogue.category_to_cids[category]:
                            if id_component in search_space:
                                search_space[id_component]["Include"] = user_choice
                except:
                    continue

    # Print out what components are available in the imported search space.
    log.info("%s   Search space..." % Timestamp(None))
    text_list = []
    for id_component in search_space:
        if CustomBool(search_space[id_component]["Include"]):
            text_list.append("%s      %s" % (Timestamp(None), id_component))
    if len(text_list) == 0:
        log.warning("%s   NONE" % Timestamp(None))
    else:
        for text in text_list:
            log.info(text)

    # Remove organising keys from portions of the imported dictionary.
    # This is an initial step towards future-proofing file structure updates.
    dict_strategy = flatten_dict(specs["Strategy"])

    strategy = Strategy(in_search_space = search_space,
                        in_n_challengers = int(dict_strategy["Number of Challengers"]),
                        do_defaults = bool(CustomBool(dict_strategy["Do Defaults"])),
                        do_random = bool(CustomBool(dict_strategy["Do Random"])),
                        do_hpo = bool(CustomBool(dict_strategy["Do HPO"])),
                        in_n_samples = int(dict_strategy["Number of Samples"]),
                        in_max_hpo_concurrency = int(dict_strategy["Max HPO Concurrency"]),
                        in_frac_validation = float(dict_strategy["Validation Fraction"]),
                        in_folds_validation = int(dict_strategy["Validation Folds"]),
                        in_n_iterations = int(specs["Optimiser"]["BOHB"]["Iterations"]),
                        in_n_partitions = int(specs["Optimiser"]["BOHB"]["Partitions"]))

    return strategy