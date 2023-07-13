# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:39:37 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .settings import SystemSettings as SS

from .data_storage import DataStorage
from .data_io import DataPort, DataPortStream
from .solver import TaskSolver

import asyncio



class AutonoMachine:
    """
    A system designed to autonomously process a machine learning task.
    """
    
    def __init__(self):
        log.info("%s - An AutonoMachine has been initialised." % Timestamp())
        
        self.data_storage = DataStorage()
        self.data_ports = dict()
        
        self.task_solver = None
        
        self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
        
        self.ops = None
        
        self.is_running = False
        self.run()
        
    def run(self):
        log.info("%s - The AutonoMachine is now running." % Timestamp())
        self.is_running = True
        
        # Check the Python environment for an asynchronous event loop.
        # Gather operations and hand them to a new/existing event loop.
        loop = asyncio.get_event_loop()
        if loop.is_running() == False:
            log.debug(("No asyncio event loop is currently running.\n"
                       "One will be launched for AutonoML operations."))
            asyncio.run(self.gather_ops())
        else:
            log.debug(("The Python environment is already running an asyncio event loop.\n"
                       "It will be used for AutonoML operations."))
            loop.create_task(self.gather_ops())
            
    async def gather_ops(self):
        self.ops = [asyncio.create_task(op) for op in [self.check_issues()]]
        await asyncio.gather(*self.ops, return_exceptions=True)
            
    # TODO: Perhaps convert to a del. Distinguish between pause and del.
    # TODO: Work out whether all asyncio tasks genuinely are stopping and why not.
    def stop(self):
        log.info("%s - The AutonoMachine is now stopping." % Timestamp())
        self.is_running = False
        
        # Cancel all asynchronous operations.
        if self.ops:
            for op in self.ops:
                op.cancel()
                
        # Stop the task solver.
        if self.task_solver:
            self.task_solver.stop()
                
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
        log.info("The AutonoMachine has %i DataPorts: %s" 
                 % (len(self.data_ports), ", ".join(self.data_ports.keys())))
        self.data_storage.info()
        
    def info_solver(self):
        """
        Utility method to give user info about the task solver and its models.
        """
        if self.task_solver:
            self.task_solver.info()
        else:
            log.error("%s - The AutonoMachine has not been given a task to solve." % Timestamp())
        
    # # TODO: Update for queries.
    # def update_storage(self, in_keys_port, in_keys_storage):
    #     """
    #     Redirects where DataPorts send their received data.
    #     Renames the lists in DataStorage, merging if required.
    #     """
    #     if not self.task_solver:
    #         self.data_storage.update(in_keys_port, in_keys_storage)
    #     else:
    #         # TODO: Relax this constraint eventually.
    #         log.error("%s - DataStorage cannot be updated while a TaskSolver exists." % Timestamp())
        
    def learn(self, in_key_target, in_keys_features = None, do_exclude = False):
        self.task_solver = TaskSolver(in_data_storage = self.data_storage,
                                      in_key_target = in_key_target, 
                                      in_keys_features = in_keys_features,
                                      do_exclude = do_exclude)
        
    # # TODO: Decide on a stop event when UI gets fleshed out.
    # async def check_stop(self):
    #     while self.is_running:
    #         await asyncio.sleep(10)
    #         # self.stop()
    
    async def check_issues(self):
        id_issue = 0
        while self.is_running:
            await asyncio.sleep(self.delay_for_issue_check)
            # TODO: Consider enums for issues.
            if id_issue in [0, 1] and not self.data_ports:
                self.warn_issue("No data ports have been assigned to the AutonoMachine.")
                id_issue = 1
            elif id_issue in [0, 2] and not self.task_solver:
                self.warn_issue("The AutonoMachine has not been given a learning task.")
                id_issue = 2
            else:
                self.delay_for_issue_check = SS.BASE_DELAY_FOR_ISSUE_CHECK
                id_issue = 0
            
            if not id_issue == 0:
                self.delay_for_issue_check *= 2
                
    def warn_issue(self, in_message):
        log.warning("%s - %i+ seconds since last check - %s" 
                    % (Timestamp(), self.delay_for_issue_check, in_message))