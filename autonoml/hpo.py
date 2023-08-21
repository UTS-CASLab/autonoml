# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:29:42 2023

@author: David J. Kedziora
"""

import logging
logging.basicConfig(level=logging.WARNING)

import ConfigSpace as CS

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import Worker, MyWorker

min_budget = 9
max_budget = 243
n_iterations = 4


class HPOWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pipeline



    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        # keys_features = in_info_process["keys_features"]
        # key_target = in_info_process["key_target"]
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

        metric =0

        return {"loss": 1 - metric, "info": None}
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()


run_id = "optimise"
name_server_host = "127.0.0.1"
name_server_port = None

# Start a name server that manages concurrent running workers across all possible threads.
name_server = hpns.NameServer(run_id = run_id, host = name_server_host, port = name_server_port)
name_server.start()

# Start a worker attached to the name server that runs in the background.
# It waits for hyperparameter configurations to evaluate.
worker = MyWorker(sleep_interval = 0, nameserver = name_server_host, run_id = run_id)
worker.run(background=True)

# Create the optimiser and start the run.
optimiser = BOHB(configspace = worker.get_configspace(),
                 run_id = run_id, nameserver = name_server_host,
                 min_budget = min_budget, max_budget = max_budget)
result = optimiser.run(n_iterations = n_iterations)

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

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print('A total of %i runs where executed.' % len(result.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in result.get_all_runs()])/max_budget))