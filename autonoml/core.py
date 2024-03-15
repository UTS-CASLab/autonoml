# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, identify_exception
from .settings import SystemSettings as SS
from .concurrency import (create_async_task_from_sync, create_async_task, 
                          schedule_this, be_patient)

from .data_storage import DataStorage
from .data_io import DataPort, DataPortStream
from .solver import ProblemSolver
from .solution import AllocationMethod, ProblemSolverInstructions
from .plot import plot_figures

from .strategy import Strategy

from typing import Dict, List, Tuple, Union

import asyncio
import multiprocess as mp
from enum import Enum



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """

    count = 0
    
    def __init__(self, do_mp = False, n_procs = None):
        self.name = "Autono_" + str(AutonoMachine.count)
        AutonoMachine.count += 1
        log.info("%s - Initialising AutonoMachine '%s'." % (Timestamp(), self.name))

        self.do_mp = do_mp
        
        if self.do_mp:
            text_concurrency = "multiprocessing"
        else:
            text_concurrency = "multithreading"
        if n_procs is None:
            n_procs = mp.cpu_count() - 1
        log.info("%s   Leveraging %i out of %i processors for %s." 
                 % (Timestamp(None), n_procs, mp.cpu_count(), text_concurrency))        
        self.n_procs = n_procs

        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.solver = None
        
        self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
        
        self.ops = None
        
        self.is_running = False     # TODO: Decide whether this variable is useful.
        self.run()

    def __del__(self):
        log.debug("Finalising Autonomachine '%s'." % self.name)

    def run(self):
        log.info("%s - AutonoMachine '%s' is now running." % (Timestamp(), self.name))
        self.is_running = True
        create_async_task_from_sync(self.gather_ops)
            
    async def gather_ops(self):
        self.ops = [create_async_task(self.check_issues)]
        group = asyncio.gather(*self.ops)
        try:
            await group
        except Exception as e:
            text_alert = ("%s - AutonoMachine '%s' encountered an error. "
                          "Cancelling Asyncio operations." % (Timestamp(), self.name))
            identify_exception(e, text_alert)
            for op in self.ops:
                op.cancel()
                
        self.is_running = False
            
    def ingest_file(self, in_filepath, in_tags: Dict[str, str] = None, in_n_instances: int = None):
        """
        Take in a .csv file and convert its contents into data to learn from, i.e. observations.

        The data can be assigned optional tags, e.g. {"source":"wiki_1", "context":"exp_1"}.
        The tags partition data within storage.
        They can be included/excluded from learning and allocated to different learners.
        """

        log.info("%s - Scheduling request for AutonoMachine '%s' to ingest data file: %s" 
                 % (Timestamp(), self.name, in_filepath))
        ref = DataPort(in_data_storage = self.data_storage, in_tags = in_tags)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(ref.ingest_file, in_filepath, in_n_instances = in_n_instances)
        
    def query_with_file(self, in_filepath, in_tags: Dict[str, str] = None, in_n_instances: int = None):
        """
        Take in a .csv file and convert its contents into data to respond to, i.e. queries.

        The data can be assigned optional tags, e.g. {"source":"wiki_1", "context":"exp_1"}.
        The tags partition data within storage.
        """

        log.info("%s - Scheduling request to query AutonoMachine '%s' with file: %s" 
                 % (Timestamp(), self.name, in_filepath))
        ref = DataPort(in_data_storage = self.data_storage, in_tags = in_tags,
                       is_for_queries = True)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(ref.ingest_file, in_filepath, in_n_instances = in_n_instances)
        
    def ingest_stream(self, in_hostname = SS.DEFAULT_HOSTNAME, in_port: int = SS.DEFAULT_PORT_OBSERVATIONS,
                      in_field_names: List[str] = None,
                      in_id_stream: str = None, in_tags: Dict[str, str] = None):
        """
        Take in a hostname and port number at which streamed data is being broadcasted.
        Treat these as observations to learn from.
        Provide names for each field and begin storing immediately or set them later and toggle storage then.

        The data can be assigned optional tags, e.g. {"source":"wiki_1", "context":"exp_1"}.
        The tags partition data within storage.
        They can be included/excluded from learning and allocated to different learners.
        """
        ref = DataPortStream(in_data_storage = self.data_storage,
                             in_hostname = in_hostname,
                             in_port = in_port,
                             in_field_names = in_field_names,
                             in_id_stream = in_id_stream,
                             in_tags = in_tags)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(ref.run_connection)

        return ref
    
    def query_with_stream(self, in_hostname = SS.DEFAULT_HOSTNAME, in_port: int = SS.DEFAULT_PORT_OBSERVATIONS,
                          in_field_names: List[str] = None,
                          in_id_stream: str = None, in_tags: Dict[str, str] = None):
        """
        Take in a hostname and port number at which streamed data is being broadcasted.
        Treat these as queries to respond to.
        Provide names for each field and begin storing immediately or set them later and toggle storage then.

        The data can be assigned optional tags, e.g. {"source":"wiki_1", "context":"exp_1"}.
        The tags partition data within storage.
        """
        ref = DataPortStream(in_data_storage = self.data_storage,
                             in_hostname = in_hostname,
                             in_port = in_port,
                             in_field_names = in_field_names,
                             in_id_stream = in_id_stream,
                             in_tags = in_tags,
                             is_for_queries = True)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(ref.run_connection)

        return ref
        

    @schedule_this
    async def info_storage(self):
        """
        Utility method to give user info about data ports and storage.
        """
        log.info("AutonoMachine '%s' has %i DataPort(s): '%s'" 
                 % (self.name, len(self.data_ports), "', '".join(self.data_ports.keys())))
        self.data_storage.info()
        
    def info_solver(self):
        """
        Utility method to give user info about the task solver and its models.
        """
        if self.solver:
            log.info("%s - Scheduling request to inspect ProblemSolver '%s' "
                     "for AutonoMachine '%s'." % (Timestamp(), self.solver.name, self.name))
            future = create_async_task_from_sync(self.solver.get_figures)
            figs = future.result()
            plot_figures(figs)
        else:
            log.error("%s - AutonoMachine '%s' has not been given a task to solve." % (Timestamp(), self.name))
        
    # # TODO: Update for queries.
    # def update_storage(self, in_keys_port, in_keys_storage):
    #     """
    #     Redirects where DataPorts send their received data.
    #     Renames the lists in DataStorage, merging if required.
    #     """
    #     if not self.solver:
    #         self.data_storage.update(in_keys_port, in_keys_storage)
    #     else:
    #         # TODO: Relax this constraint eventually.
    #         log.error("%s - DataStorage cannot be updated while a ProblemSolver exists." % Timestamp())
        
    def learn(self, in_key_target: str, in_keys_features = None, do_exclude: bool = False,
              do_immediate_responses: bool = True, 
              in_tags_allocation: List[Union[str, Tuple[str, AllocationMethod]]] = None,
              in_strategy: Strategy = None,
              in_directory_results: str = None,

              in_directory_import: str = None,
              in_import_allocation: Dict[str, Union[Tuple[str, str], Tuple[str, str, AllocationMethod],
                                                    List[Union[Tuple[str, str], Tuple[str, str, AllocationMethod]]]]] = None,
              do_compare_adaptation: bool = False,
              do_adapt_to_everything: bool = False,
              do_rerank_learners: bool = True):
        """
        Create a solver that will attempt to learn a relation between data features and a target.
        If feature keys are not provided, the relation will include every feature currently present in data storage.
        Similarly, if provided feature keys are excluded, only currently stored features will be included.

        If attempting traditional ML, with training then testing, consider disallowing immediate responses.
        Otherwise, by default, queries will be responded to ASAP after they arrive in storage, even if no learners exist yet.

        If partitions of data have been assigned tags, they can be marked here for allocation.
        For example: ["source", ("context", AllocationMethod.LEAVE_ONE_OUT)]
        This allows solution learner groups to train on differently tagged subsets of data.
        For example, each group trains on data subsets marked with one tag of "source" and any-but-one tag of "context".

        Note that a directory path can be provided to identify where results are exported to.

        Import capabilities...
        A directory path of previously exported pipelines can be provided to import them into a solver.
        An allocation dictionary can additionally specify what data subsets they adapt to.
        For example: {"pipe_1": ("source", "one"),
                      "pipe_2": [("source", "two"), ("context", "alpha", AllocationMethod.LEAVE_ONE_OUT)]}
        In this example, any pipelines with "pipe_1" in their filename will adapt on data subsets with a "source" tag of "one".
        Likewise, "pipe_2" pipelines will adapt for a "source" tag of "two", but not on data subsets with "context" tag "alpha".
        If a user desires to compare adaptation, each imported pipeline will be cloned into an adaptable and non-adaptable duo.

        Normally, adaptation only applies to the last data instance within a new batch received.
        This prevents being overwhelmed by data that continuously arrives too quickly to deal with.
        However, enable adaptation to everything if every data instance is crucial.

        Normally, learners will be reranked based on performance; disable reranking for easier external analysis.
        Warning: Do not disable reranking if caring about ensembled performance or generating new challengers.
        """
        instructions = ProblemSolverInstructions(in_key_target = in_key_target,
                                                 in_keys_features = in_keys_features,
                                                 do_exclude = do_exclude,
                                                 do_immediate_responses = do_immediate_responses,
                                                 in_tags_allocation = in_tags_allocation,
                                                 in_directory_import = in_directory_import,
                                                 in_import_allocation = in_import_allocation,
                                                 do_compare_adaptation = do_compare_adaptation,
                                                 do_adapt_to_everything = do_adapt_to_everything,
                                                 do_rerank_learners = do_rerank_learners)
        self.solver = ProblemSolver(in_data_storage = self.data_storage,
                                    in_instructions = instructions,
                                    in_strategy = in_strategy,
                                    in_n_procs = self.n_procs,
                                    do_mp = self.do_mp,
                                    in_directory_results = in_directory_results)
        
        # If running with Python, block and wait for the secondary AutonoML thread.
        # If running with IPython, do not block. Let the user interact.
        # TODO: Maybe give the user the ability to choose when to end, e.g. with input.
        be_patient()

    class Issues(Enum):
        """
        Enumeration of issues to alert the user about.
        """
        NONE = 0
        NO_DATA_PORTS = 1
        NO_SOLVER = 2
        INACTIVE_SOLVER = 3

    async def check_issues(self):
        id_issue = self.Issues.NONE
        while True:
            await asyncio.sleep(self.delay_for_issue_check)

            # Check for an issue if none exist or it has already been identified as an issue.
            if (id_issue in [self.Issues.NONE, self.Issues.NO_DATA_PORTS] 
                and not self.data_ports):
                self.warn_issue("No data ports have been assigned to AutonoMachine '%s'." % self.name)
                id_issue = self.Issues.NO_DATA_PORTS
            elif (id_issue in [self.Issues.NONE, self.Issues.NO_SOLVER] 
                  and self.solver is None):
                self.warn_issue("AutonoMachine '%s' has not been given a learning task." % self.name)
                id_issue = self.Issues.NO_SOLVER
            elif (id_issue in [self.Issues.NONE, self.Issues.INACTIVE_SOLVER] 
                  and self.solver and not self.solver.is_running):
                self.warn_issue("AutonoMachine '%s' is no longer "
                                "running its learning task." % self.name)
                id_issue = self.Issues.INACTIVE_SOLVER

            # If there are no issues, keep the check delay small.
            else:
                self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
                id_issue = self.Issues.NONE
            
            # If there is an issue, provide alerts that are less and less frequent.
            if not id_issue == 0:
                self.delay_for_issue_check *= 2
                
    def warn_issue(self, in_message):
        log.warning("%s - %i+ seconds since last check - %s" 
                    % (Timestamp(), self.delay_for_issue_check, in_message))