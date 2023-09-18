# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, identify_error
from .concurrency import create_async_task_from_sync, create_async_task, inspect_loop, loop_autonoml
from .pipeline import MLPipeline, train_pipeline
from .hpo import HPOInstructions, run_hpo, add_hpo_worker
from .strategy import Strategy, create_pipeline_random

from .data_storage import DataStorage

import __main__
import asyncio
import concurrent.futures
# import dill
# import multiprocess as mp
from copy import deepcopy

class ProblemSolverInstructions:
    def __init__(self, in_key_target: str, in_keys_features = None, do_exclude: bool = False, 
                 in_strategy: Strategy = None):
        
        self.key_target = in_key_target
        self.keys_features = in_keys_features
        self.do_exclude = do_exclude

        self.do_query_after_complete = True

        # Create a default strategy if user did not provide one.
        if in_strategy is None:
            self.strategy = Strategy()
        else:
            self.strategy = in_strategy

        # if in_strategy is None:
        #     self.do_hpo = False
        #     self.search_space = None

        #     self.do_custom = False
        # else:
        #     self.do_hpo = in_strategy.do_hpo
        #     self.search_space = in_strategy.search_space

        #     self.do_custom = in_strategy.do_custom

class ProblemSolver:
    """
    A manager for pipelines and components that learn from data and respond to queries.
    """

    count = 0
    
    def __init__(self, in_data_storage: DataStorage, 
                 in_instructions: ProblemSolverInstructions, 
                 in_n_procs: int, do_mp: bool):
        
        self.name = "Sol_" + str(ProblemSolver.count)
        ProblemSolver.count += 1
        log.info("%s - Initialising ProblemSolver '%s'." % (Timestamp(), self.name))

        # Keep a copy of the instructions that drive this ProblemSolver.
        self.instructions = in_instructions

        # Note whether to use multiprocessing and how many processors are available.
        self.do_mp = do_mp
        self.n_procs = in_n_procs

        # Store a reference to the DataStorage in the AutonoMachine.
        self.data_storage = in_data_storage

        self.key_target = None
        self.keys_features = None
        
        self.pipelines = dict()         # MLPipelines that are in production.
        self.queue_dev = None           # MLPipelines and HPO runs that are in development.
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
        
        # Instantiate the development queue now that this code is running internally within an event loop.
        self.queue_dev = asyncio.Queue()

        strategy = self.instructions.strategy

        if strategy.do_hpo:
            if len(strategy.search_space.list_predictors()) > 0:
                await self.queue_dev.put(HPOInstructions(in_strategy = strategy))
            else:
                text_warning = ("The Strategy for ProblemSolver '%s' does not suggest "
                                "any predictors in its search space.") % self.name
                log.warning("%s - %s" % (Timestamp(), text_warning))
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running HPO.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        # TODO: Do something genuine with custom pipelines.
        if strategy.do_custom:
            for count_pipeline in range(1):
                await self.add_pipeline(create_pipeline_random(in_keys_features = self.keys_features,
                                                               in_key_target = self.key_target))
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running custom pipelines.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        if self.queue_dev.qsize() == 0:
            # TODO: Perhaps wrap this up in a utils error function.
            text_error = ("ProblemSolver '%s' cannot continue. "
                          "Its Strategy does not suggest any pipelines.") % self.name
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)

        # Once instructions are processed, begin the problem solving.
        self.run()

    def run(self):
        log.info("%s - ProblemSolver '%s' is now running." % (Timestamp(), self.name))
        self.is_running = True
        create_async_task_from_sync(self.gather_ops)
        
    async def gather_ops(self):
        self.ops = list()
        # self.ops.append(create_async_task(self.process_hpo))
        self.ops.append(create_async_task(self.process_pipelines))
        self.ops.extend([create_async_task(self.process_strategy),
                         create_async_task(self.process_queries)])
        group = asyncio.gather(*self.ops)
        try:
            await group
        except Exception as e:
            text_alert = ("%s - ProblemSolver '%s' encountered an error. "
                          "Cancelling Asyncio operations." % (Timestamp(), self.name))
            identify_error(e, text_alert)
            for op in self.ops:
                op.cancel()

        self.is_running = False

    async def add_pipeline(self, in_pipeline):
        await self.queue_dev.put(in_pipeline)

    async def process_pipelines(self):
        if self.do_mp:
            executor_class = concurrent.futures.ProcessPoolExecutor
            text_executor = "processes"
            # Note: Done because subprocesses rebuilding the logger need a reference to the calling script.
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
            text_executor = "threads"

        log.info("%s - ProblemSolver '%s' has launched a pool of %i %s "
                "to train MLPipelines." % (Timestamp(), self.name, self.n_procs, text_executor))
        loop = asyncio.get_event_loop()

        # Leave one processor free when running HPO for training specified pipelines.
        # TODO: Consider whether to block subsequent HPO requests if one is already running.
        n_procs_hpo = self.n_procs - 1

        with executor_class(max_workers = self.n_procs) as executor:
            while True:
                object_dev = await self.queue_dev.get()

                idx_start = 0
                idx_end = None
                if idx_end is None:
                    idx_end = self.data_storage.observations.get_amount()

                info_process = {"keys_features": self.keys_features,
                                "key_target": self.key_target,
                                "idx_start": idx_start,
                                "idx_end": idx_end}

                if isinstance(object_dev, MLPipeline):
                    future_pipeline = loop.run_in_executor(executor, train_pipeline, object_dev,
                                                           self.data_storage.observations, info_process)
                elif isinstance(object_dev, HPOInstructions):
                    future_pipeline = loop.run_in_executor(executor, run_hpo, object_dev,
                                                           self.data_storage.observations, info_process, 
                                                           n_procs_hpo)
                    for idx_worker in range(1, n_procs_hpo):
                        loop.run_in_executor(executor, add_hpo_worker, object_dev,
                                             self.data_storage.observations, info_process, idx_worker)
                create_async_task(self.push_to_production, future_pipeline)

    async def push_to_production(self, in_future_pipeline, from_hpo = False):
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
                        Timestamp(None), pipeline.components_as_string(do_hpars = True),
                        Timestamp(None), duration_prep,
                        Timestamp(None), duration_proc,
                        Timestamp(None), metric,
                        Timestamp(None), y_pred_last, y_last))
            
            self.pipelines[pipeline.name] = pipeline
            log.info("%s   Pipeline '%s' is pushed to production." % (Timestamp(None), pipeline.name))
        except Exception as e:
            text_alert = "%s - ProblemSolver '%s' failed to process an MLPipeline." % (Timestamp(), self.name)
            identify_error(e, text_alert)
        finally:
            self.queue_dev.task_done()

    async def process_strategy(self):
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

                    # await self.queue_dev.put(pipeline)

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
                await self.queue_dev.join()
                # await self.queue_hpo.join()

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
                            Timestamp(None), pipeline.components_as_string(do_hpars = True),
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