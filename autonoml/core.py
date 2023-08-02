# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS
from .concurrency import asyncio_task_from_method

from .data_storage import DataStorage
from .data_io import DataPort, DataPortStream
from .solver import TaskSolver

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
        self.semaphore = mp.Semaphore(n_procs)

        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.solver = None
        
        self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
        
        self.ops = None
        
        self.is_running = False
        self.run()

    def __del__(self):
        log.debug("Finalising Autonomachine '%s'." % self.name)

        # # Cancel all asynchronous operations.
        # if self.ops:
        #     for op in self.ops:
        #         if not op.done():
        #             op.cancel()

    # def cleanup(self):
    #     log.info("%s - The AutonoMachine is now shutting down." % Timestamp())

    #     # Cancel all asynchronous operations.
    #     if self.ops:
    #         for op in self.ops:
    #             op.cancel()
        
    def run_asyncio(self):
        """
        This function needs to be run in a separate thread.
        """
        asyncio.run(self.gather_ops())

    def run(self):
        log.info("%s - AutonoMachine '%s' is now running." % (Timestamp(), self.name))
        self.is_running = True
        
        # # Check the Python environment for an asynchronous event loop.
        # # Gather operations and hand them to a new/existing event loop.
        # loop = asyncio.get_event_loop()
        # if loop.is_running() == False:
        #     log.debug(("No asyncio event loop is currently running.\n"
        #                "One will be launched for AutonoML operations."))
        #     thread_asyncio = threading.Thread(target=self.run_asyncio)
        #     thread_asyncio.start()
        #     # asyncio.run(self.gather_ops())
        #     #HOW TO ADD TASKS TO IT?
        # else:
        #     log.debug(("The Python environment is already running an asyncio event loop.\n"
        #                "It will be used for AutonoML operations."))
        
        asyncio_task_from_method(self.gather_ops)
            
    async def gather_ops(self):
        # self.ops = [asyncio_task_from_method(op) for op in [self.check_issues]]
        # await asyncio.gather(*self.ops)
        #, return_exceptions=True)

        self.ops = [asyncio_task_from_method(op) for op in [self.check_issues]]
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
            
    # TODO: Perhaps convert to a del. Distinguish between pause and del.
    # TODO: Review cleanup.
    def stop(self):
        log.info("%s - AutonoMachine '%s' is now stopping." % (Timestamp(), self.name))
        self.is_running = False
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
                
        # Stop the task solver.
        if self.solver:
            self.solver.stop()
                
        # Close all data ports.
        for id in list(self.data_ports.keys()):
            del self.data_ports[id]
            
    def ingest_file(self, in_filepath):
        id_data_port = str(len(self.data_ports))
        self.data_ports[id_data_port] = DataPort(in_id = id_data_port,
                                                 in_data_storage = self.data_storage)
        self.data_ports[id_data_port].ingest_file(in_filepath)
        
    def query_with_file(self, in_filepath):
        id_data_port = str(len(self.data_ports))
        self.data_ports[id_data_port] = DataPort(in_id = id_data_port,
                                                 in_data_storage = self.data_storage)
        self.data_ports[id_data_port].ingest_file(in_filepath, as_query = True)
        
    def ingest_stream(self, in_hostname, in_port):
        self.open_data_port(in_hostname = in_hostname, in_port = in_port)
                
    def open_data_port(self, in_hostname = SS.DEFAULT_HOSTNAME, in_port = SS.DEFAULT_PORT_DATA,
                       in_id = None):
        id_data_port = str(len(self.data_ports))
        if not in_id is None:
            id_data_port = str(in_id)
        self.data_ports[id_data_port] = DataPortStream(in_id = id_data_port,
                                                       in_data_storage = self.data_storage,
                                                       in_hostname = in_hostname,
                                                       in_port = in_port)
        
    def info_storage(self):
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
            self.solver.info()
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
    #         log.error("%s - DataStorage cannot be updated while a TaskSolver exists." % Timestamp())
        
    def learn(self, in_key_target, in_keys_features = None, do_exclude = False):
        self.solver = TaskSolver(in_data_storage = self.data_storage,
                                 in_key_target = in_key_target,
                                 in_keys_features = in_keys_features,
                                 do_exclude = do_exclude,
                                 in_semaphore = self.semaphore)
        
    # # TODO: Decide on a stop event when UI gets fleshed out.
    # async def check_stop(self):
    #     while self.is_running:
    #         await asyncio.sleep(10)
    #         # self.stop()
    

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
        while self.is_running:
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