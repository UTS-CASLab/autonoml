# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS
from .concurrency import create_async_task_from_sync, create_async_task, schedule_this

from .data_storage import DataStorage
from .data_io import DataPort, DataPortStream
from .solver import ProblemSolver, ProblemSolverInstructions
from .plot import plot_figures

import asyncio
import threading
import multiprocess as mp
from enum import Enum



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """

    count = 0
    
    def __init__(self, n_procs = None):
        self.name = "Autono_" + str(AutonoMachine.count)
        AutonoMachine.count += 1
        log.info("%s - Initialising AutonoMachine '%s'." % (Timestamp(), self.name))

        if n_procs is None:
            n_procs = mp.cpu_count() - 1
        log.info("%s   Leveraging %i out of %i processors." 
                 % (Timestamp(None), n_procs, mp.cpu_count()))
        # self.semaphore = mp.Semaphore(n_procs)
        self.n_procs = n_procs

        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.solver = None
        
        self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
        
        self.ops = None
        
        self.is_running = False
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
            log.error("%s - AutonoMachine '%s' encountered an error. "
                      "Cancelling Asyncio operations." % (Timestamp(), self.name))
            log.debug(e)
            for op in self.ops:
                op.cancel()
                
        self.is_running = False
            
    # # TODO: Perhaps convert to a del. Distinguish between pause and del.
    # # TODO: Review cleanup.
    # def stop(self):
    #     log.info("%s - AutonoMachine '%s' is now stopping." % (Timestamp(), self.name))
    #     self.is_running = False
        
    #     # Cancel all asynchronous operations.
    #     if self.ops:
    #         for op in self.ops:
    #             op.cancel()
                
    #     # Stop the task solver.
    #     if self.solver:
    #         self.solver.stop()
                
    #     # Close all data ports.
    #     for id in list(self.data_ports.keys()):
    #         del self.data_ports[id]
            
    def ingest_file(self, in_filepath):
        log.info("%s - Scheduling request for AutonoMachine '%s' to ingest data file: %s" 
                 % (Timestamp(), self.name, in_filepath))
        ref = DataPort(in_data_storage = self.data_storage)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(self.data_ports[ref.name].ingest_file, in_filepath)
        
    def query_with_file(self, in_filepath):
        log.info("%s - Scheduling request to query AutonoMachine '%s' with file: %s" 
                 % (Timestamp(), self.name, in_filepath))
        ref = DataPort(in_data_storage = self.data_storage)
        self.data_ports[ref.name] = ref
        create_async_task_from_sync(self.data_ports[ref.name].ingest_file, in_filepath, 
                                    as_query = True)
        
    # def ingest_stream(self, in_hostname, in_port):
    #     self.open_data_port(in_hostname = in_hostname, in_port = in_port)
                
    # def open_data_port(self, in_hostname = SS.DEFAULT_HOSTNAME, in_port = SS.DEFAULT_PORT_DATA,
    #                    in_id = None):
    #     id_data_port = str(len(self.data_ports))
    #     if not in_id is None:
    #         id_data_port = str(in_id)
    #     self.data_ports[id_data_port] = DataPortStream(in_id = id_data_port,
    #                                                    in_data_storage = self.data_storage,
    #                                                    in_hostname = in_hostname,
    #                                                    in_port = in_port)
        
    # def info_storage(self):
    #     """
    #     Utility method to give user info about data ports and storage.
    #     """
    #     log.info("%s - Scheduling request for information on data flow/storage within AutonoMachine '%s'." 
    #              % (Timestamp(), self.name))
    #     asyncio_task_from_method(self.async_info_storage,
    
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
        
    def learn(self, in_key_target, in_keys_features = None, do_exclude = False):

        instructions = ProblemSolverInstructions(in_key_target = in_key_target,
                                                 in_keys_features = in_keys_features,
                                                 do_exclude = do_exclude)
        self.solver = ProblemSolver(in_data_storage = self.data_storage,
                                    in_n_procs = self.n_procs,
                                    in_instructions = instructions)

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

            # If there are no issues, keep the checkdelay small.
            else:
                self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
                id_issue = self.Issues.NONE
            
            # If there is an issue, provide alerts that are less and less frequent.
            if not id_issue == 0:
                self.delay_for_issue_check *= 2
                
    def warn_issue(self, in_message):
        log.warning("%s - %i+ seconds since last check - %s" 
                    % (Timestamp(), self.delay_for_issue_check, in_message))