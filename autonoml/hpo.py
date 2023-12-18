# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:29:42 2023

@author: David J. Kedziora
"""

from .utils import log, DummyLogger, Timestamp, CustomBool
from .settings import SystemSettings as SS
from .pipeline import MLPipeline, train_pipeline, test_pipeline

from .hyperparameter import HPInt, HPFloat, HPCategorical
from .strategy import catalogue, Strategy, SearchSpace
from .data_storage import SharedMemoryManager
from .instructions import ProcessInformation
from .metrics import LossFunction

import ConfigSpace as CS
from copy import deepcopy
import numpy as np

from typing import List

import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

class HPOInstructions:
    """
    A class containing instructions that direct a hyperparameter optimiser.
    """
    count = 0

    def __init__(self, in_strategy: Strategy = None):
        self.name = "HPO_" + str(HPOInstructions.count)
        HPOInstructions.count += 1
        log.info("%s - Requesting HPO run '%s'." % (Timestamp(), self.name))

        self.name_server_host = "127.0.0.1"
        self.name_server_port = None

        if in_strategy is None:
            in_strategy = Strategy()

        # The fraction of data to be reserved for validation.
        # This step is done before successive halving, so quickly trained models are well-validated.
        self.frac_validation = in_strategy.frac_validation
        self.folds_validation = in_strategy.folds_validation

        # Successive 'halving' tests candidate configurations for a number of iterations.
        # Budgets, usually representing dataset size, range from minimum to maximum over the iterations.
        # Only one of the number of partitions per iteration advances.
        # The actual number of candidate tests may vary; refer to HPO package documentation for details.
        self.n_iterations = in_strategy.n_iterations
        self.n_partitions = in_strategy.n_partitions
        self.budget_min = 1/(self.n_partitions**self.n_iterations)
        self.budget_max = 1

        self.loss_function = in_strategy.loss_function

        self.search_space = in_strategy.search_space



def config_to_pipeline_structure(in_config):
    """
    Generate a pipeline from a selection made within hyperparameter space.
    """

    structure = list()
    for key_category in ["imputer", "scaler", "predictor"]:
        key_component = in_config[key_category]
        if not key_component == "":
            type_component = catalogue.components[key_component]

            config_hpars = dict()
            for key_hpar in type_component.new_hpars():
                key_full = key_component + "." + key_hpar
                if key_full in in_config.keys():
                    config_hpars[key_hpar] = in_config[key_full]
            structure.append(type_component(in_hpars = config_hpars))

    return structure

class HPOWorker(Worker):

    def __init__(self, in_data_sharer: SharedMemoryManager,
                 in_info_process: ProcessInformation, 
                 in_loss_function: LossFunction,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # # TODO: Make these only once. Do it at the solver level.
        # self.sets_training = list()
        # self.sets_validation = list()
        # set_validation, set_training = in_observations.split_by_fraction(in_fraction = in_hpo_instructions.frac_validation)
        # self.sets_training.append(set_training)
        # self.sets_validation.append(set_validation)

        # # If multiprocessing, read data efficiently from disk.
        # if not in_data_sharer is None:
        #     _, in_sets_training, in_sets_validation = in_data_sharer.load_observations()

        # self.sets_training = in_sets_training
        # self.sets_validation = in_sets_validation
        # _, self.sets_training, self.sets_validation = in_data_sharer.load_observations()
        # self.data_sharer = in_data_sharer

        # For multiprocessing, do a lot of file reading, hopefully not passing around data.
        _, sets_training, sets_validation = in_data_sharer.load_observations()
        self.sets_training, self.sets_validation = sets_training, sets_validation
        self.info_process = deepcopy(in_info_process)
        self.loss_function = in_loss_function

    # TODO: Consider the divide by zero runtime warnings generated seemingly when using categoricals.
    # TODO: Consider folding time taken into the metric.
    # TODO: Consider if stopping process is possible if takes too long.
    def compute(self, config, budget, **kwargs):

        keys_features = self.info_process.keys_features
        key_target = self.info_process.key_target
        
        losses = list()

        # # For multiprocessing, do a lot of file reading, hopefully not passing around data.
        # _, sets_training, sets_validation = self.data_sharer.load_observations()

        sets_training, sets_validation = self.sets_training, self.sets_validation

        for set_training, set_validation in zip(sets_training, sets_validation):

            # TODO: Maybe ID the test names according to configuration number.
            pipeline = MLPipeline(in_name = "Test",
                                in_keys_features = keys_features, in_key_target = key_target, do_increment_count = False,
                                in_components = config_to_pipeline_structure(in_config = config),
                                in_loss_function = self.loss_function)

            # print("Training Size: %i" % int(budget*set_training.get_amount()))
            pipeline, _, _ = train_pipeline(in_pipeline = pipeline,
                                            in_data_collection = set_training,
                                            in_info_process = self.info_process,
                                            in_frac_data = budget)
            
            # info = "\ntrain_prep %s" % self.info_process.duration_prep
            # info += "\ntrain_proc %s" % self.info_process.duration_proc
            
            # print("Validation Size: %i" % set_validation.get_amount())
            pipeline, _, _ = test_pipeline(in_pipeline = pipeline,
                                           in_data_collection = set_validation,
                                           in_info_process = self.info_process)

            # info += "\nvalid_prep %s" % self.info_process.duration_prep
            # info += "\nvalid_proc %s" % self.info_process.duration_proc
            
            losses.append(pipeline.get_loss())

        if len(losses) == 0:
            loss = np.inf
        else:
            loss = sum(losses)/len(losses)
        # print("Loss: %f" % loss)

        # TODO: Consider more informative info.
        return {"loss": loss, "info": None}
    
    # TODO: Make a setting for empty-string magic values.
    @staticmethod
    def get_configspace(in_search_space: SearchSpace):

        cs = CS.ConfigurationSpace()

        imputers = in_search_space.list_imputers()
        scalers = in_search_space.list_scalers() + [""]
        predictors = in_search_space.list_predictors()

        if len(imputers) == 0: imputers = [""]

        # Check whether to include any associated hyperparameters in the config space.
        for tuple_category in [("imputer", imputers), ("scaler", scalers), ("predictor", predictors)]:
            hp_cat = CS.CategoricalHyperparameter(tuple_category[0], tuple_category[1])
            cs.add_hyperparameter(hp_cat)

            for id_component in tuple_category[1]:
                if id_component == "":
                    break
                if "Hpars" in in_search_space[id_component]:
                    dict_hpars = in_search_space[id_component]["Hpars"]
                    for name_hpar in dict_hpars:
                        do_vary = CustomBool(dict_hpars[name_hpar]["Vary"])

                        if do_vary:
                            # Copy the appropriate hyperparameter and update it as desired.
                            hpar = deepcopy(catalogue.components[id_component].new_hpars()[name_hpar])
                            hpar.from_dict_config(dict_hpars[name_hpar])

                            key_full = id_component + "." + name_hpar

                            # Create the right config-space hyperparameter.
                            if isinstance(hpar, HPInt):
                                hp = CS.UniformIntegerHyperparameter(key_full,
                                                                     lower = hpar.min,
                                                                     upper = hpar.max,
                                                                     default_value = hpar.default,
                                                                     log = hpar.is_log_scale)
                            elif isinstance(hpar, HPFloat):
                                hp = CS.UniformFloatHyperparameter(key_full,
                                                                   lower = hpar.min,
                                                                   upper = hpar.max,
                                                                   default_value = hpar.default,
                                                                   log = hpar.is_log_scale)
                            elif isinstance(hpar, HPCategorical):
                                hp = CS.CategoricalHyperparameter(key_full,
                                                                  choices = hpar.options,
                                                                  default_value = hpar.default)
                            else:
                                # TODO: Make this error more informative.
                                raise NotImplementedError
                            
                            # Use the hyperparameter if the right predictor is being used.
                            cs.add_hyperparameter(hp)
                            cond = CS.EqualsCondition(hp, hp_cat, id_component)
                            cs.add_condition(cond)

        return(cs)
    


def create_pipelines_default(in_keys_features: List[str], in_key_target: str, in_strategy: Strategy):

    if in_strategy is None:
        search_space = SearchSpace()
    else:
        search_space = in_strategy.search_space

    cs = HPOWorker.get_configspace(search_space)
    imputer_names = cs.get_hyperparameter("imputer").choices
    scaler_names = cs.get_hyperparameter("scaler").choices
    predictor_names = cs.get_hyperparameter("predictor").choices

    pipelines = list()
    for predictor_name in predictor_names:

        # Include a default imputer and scaler if they exist.
        structure = list()
        if not imputer_names[0] == "":
            structure.append(catalogue.components[imputer_names[0]]())
        if not scaler_names[0] == "":
            structure.append(catalogue.components[scaler_names[0]]())

        pipeline = MLPipeline(in_keys_features = in_keys_features, in_key_target = in_key_target,
                              in_components = structure + [catalogue.components[predictor_name]()],
                              in_loss_function = in_strategy.loss_function)
        pipelines.append(pipeline)
    
    return pipelines

def create_pipeline_random(in_keys_features, in_key_target, in_strategy):

    if in_strategy is None:
        search_space = SearchSpace()
    else:
        search_space = in_strategy.search_space

    cs = HPOWorker.get_configspace(search_space)
    config = cs.sample_configuration()

    pipeline = MLPipeline(in_keys_features = in_keys_features, in_key_target = in_key_target,
                          in_components = config_to_pipeline_structure(in_config = config),
                          in_loss_function = in_strategy.loss_function)
    
    return pipeline



def add_hpo_worker(in_hpo_instructions: HPOInstructions,
                   in_data_sharer: SharedMemoryManager,
                   in_info_process: ProcessInformation, in_idx: int, do_background: bool = False):
    """
    Supplements a current HPO run with an additional worker.
    The worker should terminate when the name server is done with the run.
    This is designed for multiprocessing.
    """

    run_id = in_hpo_instructions.name
    name_server_host = in_hpo_instructions.name_server_host

    if SS.LOG_HPO_WORKER:
        log_to_use = log
    else:
        log_to_use = DummyLogger()

    worker = HPOWorker(in_data_sharer = in_data_sharer,
                       in_info_process = in_info_process,
                       in_loss_function = in_hpo_instructions.loss_function,
                       nameserver = name_server_host, run_id = run_id, id = in_idx, logger = log_to_use)
    try:
        worker.run(background = do_background)
        return worker
    except Exception as e:
        return e

def run_hpo(in_hpo_instructions: HPOInstructions,
            in_data_sharer: SharedMemoryManager,
            in_info_process: ProcessInformation):
    """
    Runs a hyperparameter optimisation according to specified instructions.
    During this run, configured pipelines are trained on a set of provided observations.

    In detail, a name server is first set up to manage a specified number of workers.
    This function even constructs the first HPO worker to operate in the background.
    However, the other workers must be constructed via another function.
    """

    run_id = in_hpo_instructions.name
    name_server_host = in_hpo_instructions.name_server_host
    name_server_port = in_hpo_instructions.name_server_port
    n_partitions = in_hpo_instructions.n_partitions
    n_iterations = in_hpo_instructions.n_iterations
    budget_min = in_hpo_instructions.budget_min
    budget_max = in_hpo_instructions.budget_max
    search_space = in_hpo_instructions.search_space

    # Start a name server that manages concurrent running workers across all possible processes/threads.
    name_server = hpns.NameServer(run_id = run_id, host = name_server_host, port = name_server_port)
    name_server.start()

    # The main thread/process can run a worker in the background around the name server.
    # Note: No need to pass a data-sharer as the data has just been read from file.
    worker = add_hpo_worker(in_hpo_instructions = in_hpo_instructions, in_data_sharer = in_data_sharer,
                            in_info_process = in_info_process, in_idx = 0, do_background = True)

    # extra = "\n" + str(Timestamp()) + " Start HPO"

    # Create the optimiser and start the run.
    if SS.LOG_HPO_OPTIMISER:
        log_to_use = log
    else:
        log_to_use = DummyLogger()

    optimiser = BOHB(configspace = worker.get_configspace(in_search_space = search_space),
                     run_id = run_id, nameserver = name_server_host, logger = log_to_use,
                     min_budget = budget_min, max_budget = budget_max, eta = n_partitions)
    result = optimiser.run(n_iterations = n_iterations)

    # extra += "\n" + str(Timestamp()) + " End HPO"

    # Shutdown the optimiser and name server once complete.
    optimiser.shutdown(shutdown_workers = True)
    name_server.shutdown()

    # Each optimiser returns a hpbandster.core.result.Result object.
    # It holds information about the run like the incumbent configuration.
    # TODO: Deal with failed HPO, e.g. when id_best is None.
    all_runs = result.get_all_runs()
    id_to_config = result.get_id2config_mapping()

    id_best = result.get_incumbent_id()
    run_best = result.get_runs_by_id(id_best)[-1]

    config_best = id_to_config[id_best]["config"]
    loss_best = run_best.loss
    _ = run_best.info   # TODO: Do something with the information.
    # info = run_best.info
    # extra += info

    text_hpo = ("%s - HPO run '%s' has sampled %i pipelines with %i unique configurations.\n"
                "%s   Equivalent fully trained pipelines based on net budget: %.3f\n"
                "%s   Best found max-budget configuration: %s\n"
                "%s   Best found max-budget validation loss: %s\n"
                "%s   Time taken to run HPO: %0.3f s"
                % (Timestamp(), run_id, len(all_runs), len(id_to_config.keys()),
                Timestamp(None), sum([run.budget for run in all_runs])/budget_max,
                Timestamp(None), config_best,
                Timestamp(None), loss_best,
                Timestamp(None), all_runs[-1].time_stamps["finished"] - all_runs[0].time_stamps["started"]))

    # text_hpo += extra

    # If multiprocessing, read data efficiently from disk.
    in_observations, _, _ = in_data_sharer.load_observations()

    # Based on the best configuration, create a new pipeline.
    keys_features = in_info_process.keys_features
    key_target = in_info_process.key_target
    pipeline = MLPipeline(in_name = "Pipe_" + run_id,
                          in_keys_features = keys_features, in_key_target = key_target,
                          in_components = config_to_pipeline_structure(in_config = config_best),
                          in_loss_function = in_hpo_instructions.loss_function)
    pipeline, _, info_process = train_pipeline(in_pipeline = pipeline,
                                               in_data_collection = in_observations,
                                               in_info_process = in_info_process)
    
    # Short of further testing, its starting loss is the validation score it received during HPO.
    pipeline.set_loss(loss_best)
    
    info_process.set_text(text_hpo)

    return pipeline, info_process