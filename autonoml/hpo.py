# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:29:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pipeline import MLPipeline, train_pipeline
from .components.river import OnlineLinearRegressor

from .data_storage import DataCollection

import time
import ConfigSpace as CS
from copy import deepcopy

import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB

class HPOInstructions:
    """
    A class containing instructions that direct a hyperparameter optimiser.
    """
    count = 0

    def __init__(self):
        self.name = "HPO_" + str(HPOInstructions.count)
        HPOInstructions.count += 1
        log.info("%s - Requesting HPO run '%s'." % (Timestamp(), self.name))

        self.name_server_host = "127.0.0.1"
        self.name_server_port = None

        # Successive 'halving' tests candidate configurations for a number of iterations.
        # Budgets, usually representing dataset size, range from minimum to maximum over the iterations.
        # Only one of the number of partitions per iteration advances.
        # The actual number of candidate tests may vary; refer to HPO package documentation for details.
        self.n_partitions = 3
        self.n_iterations = 4
        self.budget_min = 1/(self.n_partitions**self.n_iterations)
        self.budget_max = 1

def add_hpo_worker(in_hpo_instructions: HPOInstructions, in_observations: DataCollection,
                   in_info_process, in_idx: int):
    """
    Supplements a current HPO run with an additional worker.
    The worker should terminate when the name server is done with the run.
    This is designed for multiprocessing.
    """

    run_id = in_hpo_instructions.name
    name_server_host = in_hpo_instructions.name_server_host

    time.sleep(5)   # Artificial delay to ensure the name server is already running.
    worker = HPOWorker(in_observations = in_observations, in_info_process = in_info_process,
                       nameserver = name_server_host, run_id = run_id, id = in_idx, logger = log)
    worker.run(background = False)

def run_hpo(in_hpo_instructions: HPOInstructions, in_observations: DataCollection, 
            in_info_process, in_n_workers: int):
    """
    Runs a hyperparameter optimisation according to specified instructions.
    During this run, configured pipelines are trained on a set of provided observations.

    In detail, a name server is first set up to manage a specified number of workers.
    This function even constructs the first HPO worker to operate in the background.
    However, the other workers must be constructed via another function.
    If there are not enough workers, HPO will not run.
    """

    run_id = in_hpo_instructions.name
    name_server_host = in_hpo_instructions.name_server_host
    name_server_port = in_hpo_instructions.name_server_port
    n_partitions = in_hpo_instructions.n_partitions
    n_iterations = in_hpo_instructions.n_iterations
    budget_min = in_hpo_instructions.budget_min
    budget_max = in_hpo_instructions.budget_max

    # Start a name server that manages concurrent running workers across all possible processes/threads.
    name_server = hpns.NameServer(run_id = run_id, host = name_server_host, port = name_server_port)
    name_server.start()

    # # Start workers attached to the name server that runs in the background.
    # # It waits for hyperparameter configurations to evaluate.
    # if do_mp:
    #     pass
    # else:
    #     workers = list()
    #     for idx_worker in range(in_n_procs):
    #         worker = HPOWorker(in_observations = in_observations, in_info_process = in_info_process,
    #                            nameserver = name_server_host, run_id = run_id, id = idx_worker)
    #         worker.run(background = True)
    #         workers.append(worker)

    # The main thread/process can run a worker in the background around the name server.
    worker = HPOWorker(in_observations = in_observations, in_info_process = in_info_process,
                       nameserver = name_server_host, run_id = run_id, id = 0, logger = log)
    worker.run(background = True)

    # Create the optimiser and start the run.
    optimiser = BOHB(configspace = worker.get_configspace(),
                     run_id = run_id, nameserver = name_server_host, logger = log,
                     min_budget = budget_min, max_budget = budget_max, eta = n_partitions)
    result = optimiser.run(n_iterations = n_iterations, min_n_workers = in_n_workers)

    # if do_mp:
    #     result = optimiser.run(n_iterations = n_iterations, min_n_workers = in_n_procs)
    # else:
    #     result = optimiser.run(n_iterations = n_iterations)

    # Shutdown the optimiser and name server once complete.
    optimiser.shutdown(shutdown_workers = True)
    name_server.shutdown()

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    all_runs = result.get_all_runs()

    # print(id2config)

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations were sampled.' % len(id2config.keys()))
    print('A total of %i runs were executed.' % len(all_runs))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/budget_max))
    print('The run took %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    keys_features = in_info_process["keys_features"]
    key_target = in_info_process["key_target"]
    config = id2config[incumbent]["config"]
    pipeline = MLPipeline(in_name = "Pipe_" + run_id,
                          in_keys_features = keys_features, in_key_target = key_target,
                          in_components = [OnlineLinearRegressor(in_hpars = {"batch_size": config["batch_size"],
                                                                             "learning_rate": config["learning_rate"]})])
    pipeline, info_process = train_pipeline(in_pipeline = pipeline,
                                            in_observations = in_observations,
                                            in_info_process = in_info_process)

    return pipeline, in_info_process



class HPOWorker(Worker):

    def __init__(self, in_observations: DataCollection, in_info_process, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observations = in_observations
        self.info_process = deepcopy(in_info_process)   # Deepcopy as budgets will differ.



    def compute(self, config, budget, **kwargs):
        # """
        # Simple example for a compute function
        # The loss is just a the config + some noise (that decreases with the budget)

        # Args:
        #     config: dictionary containing the sampled configurations by the optimizer
        #     budget: (float) amount of time/epochs/etc. the model can use to train

        # Returns:
        #     dictionary with mandatory fields:
        #         'loss' (scalar)
        #         'info' (dict)
        # """
        print(budget)

        keys_features = self.info_process["keys_features"]
        key_target = self.info_process["key_target"]
        # idx_start = in_info_process["idx_start"]
        # idx_end = in_info_process["idx_end"]

        # time_start = Timestamp().time
        # x, y = in_observations.get_data(in_keys_features = keys_features,
        #                                 in_key_target = key_target,
        #                                 in_idx_start = idx_start,
        #                                 in_idx_end = idx_end)
        # time_end = Timestamp().time
        # duration_prep = time_end - time_start

        # time_start = Timestamp().time
        # _, metric = in_pipeline.process(x, y, do_remember = True, for_training = True)
        # time_end = Timestamp().time
        # duration_proc = time_end - time_start

        # in_info_process["metric"] = metric
        # in_info_process["duration_prep"] = duration_prep
        # in_info_process["duration_proc"] = duration_proc

        pipeline = MLPipeline(in_name = "Test",
                              in_keys_features = keys_features, in_key_target = key_target, do_increment_count = False,
                              in_components = [OnlineLinearRegressor(in_hpars = {"batch_size": config["batch_size"],
                                                                                "learning_rate": config["learning_rate"]})])

        self.info_process["fraction"] = budget
        pipeline, info_process = train_pipeline(in_pipeline = pipeline, 
                                                in_observations = self.observations, 
                                                in_info_process = self.info_process)
        metric = info_process["metric"]
        # metric = 0

        print(metric)

        return {"loss": 1 - metric, "info": info_process}
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace({
            "batch_size": CS.Integer("batch_size", default = 1, bounds = (1, 1000)),
            "learning_rate": CS.Float("learning_rate", default = 0.01, bounds = (1e-9, 1), log = True)
        })
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return(config_space)