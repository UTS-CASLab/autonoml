# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .concurrency import create_async_task_from_sync, create_async_task, inspect_loop, loop_autonoml
from .pipeline import MLPipeline, train_pipeline
from .pool import (StandardScaler,
                   PartialLeastSquaresRegressor,
                   LinearRegressor,
                   LinearSupportVectorRegressor,
                   OnlineStandardScaler,
                   OnlineLinearRegressor)

import asyncio
import concurrent.futures
# import dill
import multiprocess as mp
from copy import deepcopy

import time

# def task(in_pipeline):
#     print(in_pipeline)
#     # with in_semaphore:
#     #     print(in_pipeline.name)

class ProblemSolverInstructions:
    def __init__(self, in_key_target, in_keys_features = None, do_exclude = False):
        self.key_target = in_key_target
        self.keys_features = in_keys_features
        self.do_exclude = do_exclude

        self.do_query_after_complete = True

class ProblemSolver:
    """
    A wrapper for pipelines and components that learn from data and respond to queries.
    Stores references to attributes of an AutonoMachine:
        - DataStorage, for passing data and queries to pipelines.
        - Semaphore, for distributing operations across multiple processors.
    """

    count = 0
    
    def __init__(self, in_data_storage, in_n_procs, in_instructions):
        self.name = "Sol_" + str(ProblemSolver.count)
        ProblemSolver.count += 1
        log.info("%s - Initialising ProblemSolver '%s'." % (Timestamp(), self.name))

        # Keep a copy of the instructions that drive this ProblemSolver.
        self.instructions = in_instructions

        # Keep a reference to how many processors are available.
        self.n_procs = in_n_procs

        self.data_storage = in_data_storage

        self.key_target = None
        self.keys_features = None
        
        self.pipelines = dict()         # MLPipelines that are in production.
        self.queue_pipelines = None     # MLPipelines that are in development.
        # Note: This queue must not be instantiated as an asyncio queue until within a coroutine.
        
        # Keep track of the data-storage instance up to which model has used.
        # TODO: Consider variant starting points for the model and update log messages.
        self.idx_data = 0
        self.idx_queries = 0
        
        # # Set up a variable that can be awaited elsewhere.
        # # This 'switch', when flicked, signals that the pipelines can be queried.
        # self.can_query = asyncio.Future()
        
        self.ops = None
        self.is_running = False
        create_async_task_from_sync(self.prepare)

    def __del__(self):
        log.debug("Finalising ProblemSolver '%s'." % self.name)

    async def prepare(self):
        # Process instructions.
        key_target = self.instructions.key_target
        keys_features = self.instructions.keys_features
        do_exclude = self.instructions.do_exclude
        future = create_async_task(self.set_target_and_features,
                                   in_key_target = key_target,
                                   in_keys_features = keys_features,
                                   do_exclude = do_exclude)
        o1, o2 = await future
        self.key_target = o1
        self.keys_features = o2
        
        # Instantiate the queue now so that it is linked to the right event loop.
        self.queue_pipelines = asyncio.Queue()

        # Set up pipelines.
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [PartialLeastSquaresRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [StandardScaler(),
                                                      PartialLeastSquaresRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [LinearRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [LinearSupportVectorRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [StandardScaler(),
                                                      LinearSupportVectorRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [OnlineLinearRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [OnlineLinearRegressor(in_hpars = {"batch_size":10000})]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [OnlineStandardScaler(),
                                                      OnlineLinearRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                     in_key_target = self.key_target,
                                     in_components = [StandardScaler(),
                                                      OnlineLinearRegressor()]))
        await self.add_pipeline(MLPipeline(in_keys_features = self.keys_features,
                                           in_key_target = self.key_target,
                                           in_components = 
                                           [OnlineStandardScaler(in_hpars = {"batch_size":10}),
                                            OnlineLinearRegressor(in_hpars = {"batch_size":10})]))

        # Once instructions are processed, begin the problem solving.
        self.run()

    def run(self):
        log.info("%s - ProblemSolver '%s' is now running." % (Timestamp(), self.name))
        self.is_running = True
        create_async_task_from_sync(self.gather_ops)
        
    async def gather_ops(self):
        self.ops = [create_async_task(self.process_pipelines),
                    create_async_task(self.process_strategy), 
                    create_async_task(self.process_queries)]
        group = asyncio.gather(*self.ops)
        try:
            await group
        except Exception as e:
            log.error("%s - ProblemSolver '%s' encountered an error. "
                      "Cancelling Asyncio operations." % (Timestamp(), self.name))
            log.debug(e)
            for op in self.ops:
                op.cancel()

        self.is_running = False
        
    # def stop(self):
    #     # Cancel all asynchronous operations.
    #     if self.ops:
    #         for op in self.ops:
    #             op.cancel()

    async def add_pipeline(self, in_pipeline):
        # self.pipelines[in_pipeline.name] = in_pipeline
        await self.queue_pipelines.put(in_pipeline)

    async def process_pipelines(self):
        log.info("%s - ProblemSolver '%s' has launched a process pool "
                 "to train MLPipelines." % (Timestamp(), self.name))
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers = self.n_procs) as executor:
            while True:
                pipeline = await self.queue_pipelines.get()

                idx_start = 0
                idx_end = None
                if idx_end is None:
                    idx_end = self.data_storage.observations.get_amount()

                # time_start = Timestamp().time
                # x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
                #                                   in_key_target = self.key_target,
                #                                   in_idx_start = idx_start,
                #                                   in_idx_end = idx_end)
                # time_end = Timestamp().time
                # duration_prep = time_end - time_start

                # info_process = {"idx_start": idx_start,
                #                 "idx_end": idx_end,
                #                 "duration_prep": duration_prep}

                info_process = {"keys_features": self.keys_features,
                                "key_target": self.key_target,
                                "idx_start": idx_start,
                                "idx_end": idx_end}

                # future_pipeline = loop.run_in_executor(executor, process_pipeline, pipeline, 
                #                                        deepcopy(x), deepcopy(y), info_process)
                future_pipeline = loop.run_in_executor(executor, train_pipeline, pipeline,
                                                       self.data_storage.observations, info_process)
                create_async_task(self.push_to_production, future_pipeline)

                # # Check if anything else is waiting after submitting a pipeline for processing.
                # await asyncio.sleep(0)

    async def push_to_production(self, in_future_pipeline):
        try:
            pipeline, info_process = await in_future_pipeline

            idx_start = info_process["idx_start"]
            idx_end = info_process["idx_end"]
            duration_prep = info_process["duration_prep"]
            duration_proc = info_process["duration_proc"]
            metric = info_process["metric"]
            y_last = pipeline.training_y_true[-1]
            y_pred_last = pipeline.training_y_response[-1]

            log.info("%s - Pipeline '%s' has learned from a total of %i observations.\n"
                     "%s   Structure: %s\n"
                     "%s   Time taken to retrieve data: %.3f s\n"
                     "%s   Time taken to train/score pipeline on data: %.3f s\n"
                     "%s   Score on those observations: %f\n"
                     "%s   Last observation: Prediction '%s' vs True Value '%s'"
                     % (Timestamp(), pipeline.name, idx_end - idx_start,
                        Timestamp(None), pipeline.components_as_string(),
                        Timestamp(None), duration_prep,
                        Timestamp(None), duration_proc,
                        Timestamp(None), metric,
                        Timestamp(None), y_pred_last, y_last))
            
            self.pipelines[pipeline.name] = pipeline
            log.info("%s   Pipeline '%s' is pushed to production." % (Timestamp(None), pipeline.name))
        except Exception as e:
            log.error("%s - ProblemSolver '%s' failed to process an MLPipeline." 
                      % (Timestamp(), self.name))
            log.debug(e)
        finally:
            self.queue_pipelines.task_done()

    async def process_strategy(self):
        # self.can_query.set_result(True)

        # dev_waiting = mp.queue.Queue()
        # dev_done = mp.queue.Queue()

        while True:

            # Check for new data and learn from it.
            if self.idx_data < self.data_storage.observations.get_amount():
                self.idx_data = self.data_storage.observations.get_amount()
                
                # df = self.data_storage.get_dataframe()
                # print(df)
                # df = df.sample(frac = 1)
                # print(df)
                
                # processes = list()
                # for pipeline_key in list(self.pipelines.keys()):
                #     pipeline = self.pipelines[pipeline_key]
                #     try:
                #         time_start = Timestamp().time
                #         x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
                #                                           in_key_target = self.key_target,
                #                                           in_idx_end = 2)
                #         _, metric = pipeline.process(x, y, do_remember = True, for_training = True)
                #         time_end = Timestamp().time
                #         y_last = pipeline.training_y_true[-1]
                #         y_pred_last = pipeline.training_y_response[-1]
                
                #         log.info("%s - Pipeline '%s' has learned from a total of %i observations.\n"
                #                 "%s   Structure: %s\n"
                #                 "%s   Time taken to retrieve, learn and score pipeline on data: %.3f s\n"
                #                 "%s   Score on those observations: %f\n"
                #                 "%s   Last observation: Prediction '%s' vs True Value '%s'"
                #                 % (Timestamp(), pipeline.name, self.idx_data,
                #                     Timestamp(None), pipeline.components_as_string(),
                #                     Timestamp(None), time_end - time_start,
                #                     Timestamp(None), metric,
                #                     Timestamp(None), y_pred_last, y_last))
                #     except Exception as e:
                #         log.error("%s - Pipeline '%s' failed to process data while learning. "
                #                   "Deleting it and continuing." % (Timestamp(), pipeline_key))
                #         log.debug(e)
                #         del self.pipelines[pipeline_key]

                    # await self.queue_pipelines.put(pipeline)

                #     # EXAMINE PROCESS POOL EXECUTOR. loop.run_in_executor()
                #     print(1)
                #     process = mp.Process(target=task)#, args=(pipeline,))
                #     print(2)
                #     processes.append(process)
                #     process.start()
                #     print(3)

                # for process in processes:
                #     print("yo")
                #     process.join()

            # self.can_query.set_result(True)
                
            await self.data_storage.has_new_observations
        
        # while True:
        #     # Check for new data and learn from it.
        #     if self.idx_data < self.data_storage.observations.get_amount():
        #         self.idx_data = self.data_storage.observations.get_amount()
                
        #         # df = self.data_storage.get_dataframe()
        #         # print(df)
        #         # df = df.sample(frac = 1)
        #         # print(df)
                        
        #         for pipeline_key in list(self.pipelines.keys()):
        #             pipeline = self.pipelines[pipeline_key]
        #             try:
        #                 time_start = Timestamp().time
        #                 x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
        #                                                   in_key_target = self.key_target,
        #                                                   in_idx_end = 2)
        #                 _, metric = pipeline.process(x, y, do_remember = True, for_training = True)
        #                 time_end = Timestamp().time
        #                 y_last = pipeline.training_y_true[-1]
        #                 y_pred_last = pipeline.training_y_response[-1]
                
        #                 log.info("%s - Pipeline '%s' has learned from a total of %i observations.\n"
        #                         "%s   Structure: %s\n"
        #                         "%s   Time taken to retrieve, learn and score pipeline on data: %.3f s\n"
        #                         "%s   Score on those observations: %f\n"
        #                         "%s   Last observation: Prediction '%s' vs True Value '%s'"
        #                         % (Timestamp(), pipeline.name, self.idx_data,
        #                             Timestamp(None), pipeline.components_as_string(),
        #                             Timestamp(None), time_end - time_start,
        #                             Timestamp(None), metric,
        #                             Timestamp(None), y_pred_last, y_last))
        #             except Exception as e:
        #                 log.error("%s - Pipeline '%s' failed to process data while learning. "
        #                           "Deleting it and continuing." % (Timestamp(), pipeline_key))
        #                 log.debug(e)
        #                 del self.pipelines[pipeline_key]
                
        #     await self.data_storage.has_new_observations

    async def process_queries(self):
        
        while True:

            # If required, ensure no pipelines are awaiting development before querying.
            if self.instructions.do_query_after_complete:
                await self.queue_pipelines.join()

            # Check if there are more queries in storage than have been processed.
            # If not, wait until new queries arive.
            # TODO: Compare against a data storage index rather than a length.
            if not self.idx_queries < self.data_storage.queries.get_amount():
                await self.data_storage.has_new_queries

            # Fix how many queries to process based on what is available at the time.
            idx_stop = self.data_storage.queries.get_amount()

            # Extract the queries to process.
            x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
                                              in_key_target = self.key_target,
                                              in_idx_start = self.idx_queries,
                                              in_idx_end = idx_stop,
                                              from_queries = True)

            # Go through every pipeline in production and process the queries.
            # TODO: Ensemble them to derive a single set of responses.
            # processes = list()
            for pipeline_key in list(self.pipelines.keys()):
                try:
                    pipeline = self.pipelines[pipeline_key]
                except Exception as e:
                    log.warning("%s - Pipeline '%s' disappeared in the middle of a query phase. "
                                "Continuing to query the next pipeline." % (Timestamp(), pipeline_key))
                    log.debug(e)
                    continue

                time_start = Timestamp().time
                _, metric = pipeline.process(deepcopy(x), deepcopy(y), 
                                             do_learn = False, do_remember = True)
                time_end = Timestamp().time
                y_last = pipeline.testing_y_true[-1]
                y_pred_last = pipeline.testing_y_response[-1]
        
                log.info("%s - Pipeline '%s' has responded to a total of %i queries.\n"
                            "%s   Structure: %s\n"
                            "%s   Time taken to score pipeline on queries: %.3f s\n"
                            "%s   Score on those queries: %f\n"
                            "%s   Last query: Prediction '%s' vs True Value '%s'"
                            % (Timestamp(), pipeline.name, idx_stop - self.idx_queries,
                            Timestamp(None), pipeline.components_as_string(),
                            Timestamp(None), time_end - time_start,
                            Timestamp(None), metric,
                            Timestamp(None), y_pred_last, y_last))
                
            #     print(1)
            #     process = mp.Process(target=task, args=("woo",))
            #     print(2)
            #     processes.append(process)
            #     process.start()
            #     print(3)

            # for process in processes:
            #     process.join()
                
            # Update an index to acknowledge the queries that have been processed.
            self.idx_queries = idx_stop

            # Let other asynchronous tasks proceed if they are waiting.
            # Prevents endlessly focussing on inference if development is possible.
            await asyncio.sleep(0)

        # while True:
        #     await self.can_query
            
        #     # Check for new queries and derive responses.
        #     # Score them if possible.
        #     if self.idx_queries < self.data_storage.queries.get_amount():
        #         self.idx_queries = self.data_storage.queries.get_amount()
                
        #         # df = self.data_storage.get_dataframe()
        #         # print(df)
        #         # df = df.sample(frac = 1)
        #         # print(df)
                        
        #         for pipeline_key in list(self.pipelines.keys()):
        #             pipeline = self.pipelines[pipeline_key]

        #             time_start = Timestamp().time
        #             x, y = self.data_storage.get_data(in_keys_features = self.keys_features,
        #                                               in_key_target = self.key_target,
        #                                               from_queries = True)
        #             _, metric = pipeline.process(x, y, do_learn = False, do_remember = True)
        #             time_end = Timestamp().time
        #             y_last = pipeline.testing_y_true[-1]
        #             y_pred_last = pipeline.testing_y_response[-1]
            
        #             log.info("%s - Pipeline '%s' has responded to a total of %i queries.\n"
        #                      "%s   Structure: %s\n"
        #                      "%s   Time taken to retrieve and score pipeline on queries: %.3f s\n"
        #                      "%s   Score on those queries: %f\n"
        #                      "%s   Last query: Prediction '%s' vs True Value '%s'"
        #                      % (Timestamp(), pipeline.name, self.idx_queries,
        #                         Timestamp(None), pipeline.components_as_string(),
        #                         Timestamp(None), time_end - time_start,
        #                         Timestamp(None), metric,
        #                         Timestamp(None), y_pred_last, y_last))
            
        #     await self.data_storage.has_new_queries
            
    # TODO: Include error checking for no features. Error-check target existence somewhere too.
    # TODO: Consider making DataCollection references more robust.
    async def set_target_and_features(self, in_key_target,
                                      in_keys_features = None, do_exclude = False):
        
        if in_key_target in self.data_storage.observations.data:
            key_target = in_key_target
        else:
            text_error = "Desired target key '%s' cannot be found in DataStorage." % in_key_target 
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)
        
        keys_features = list()
        # If the user provided feature keys, but not to exclude...
        # Include them as long as such features exist in the data storage.
        if in_keys_features and not do_exclude:
            for key_feature in in_keys_features:
                if key_feature in self.data_storage.observations.data:
                    keys_features.append(key_feature)
                else:
                    log.warning("%s - Desired feature key '%s' cannot be found in DataStorage.\n"
                                "%s   The ProblemSolver will ignore it." 
                                % (Timestamp(), key_feature, Timestamp(None)))
        # Otherwise, include every feature existing in the data storage...
        # Except for feature keys specified with the intention of excluding.
        else:
            for dkey in self.data_storage.observations.data.keys():
                if not dkey == in_key_target:
                    if do_exclude and dkey in in_keys_features:
                        log.info("%s - DataStorage key '%s' has been marked as not a feature.\n"
                                 "%s   The ProblemSolver will ignore it."
                                 % (Timestamp(), dkey, Timestamp(None)))
                    else:
                        keys_features.append(dkey)

        return key_target, keys_features


    async def get_figures(self):
        """
        Utility method to get informative figures about the task solver and its models.
        """
        figs = list()
        for _, pipeline in self.pipelines.items():
            figs.extend(pipeline.inspect_structure())
            figs.extend(pipeline.inspect_performance(for_training = True))
            figs.extend(pipeline.inspect_performance(for_training = False))
        return figs