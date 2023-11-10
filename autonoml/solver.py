# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, identify_exception
from .concurrency import create_async_task_from_sync, create_async_task
from .pipeline import MLPipeline
from .hpo import HPOInstructions, run_hpo, add_hpo_worker, create_pipelines_default, create_pipeline_random
from .solution import ProblemSolverInstructions, ProblemSolution
from .solver_ops import (filter_observations, prepare_data, develop_pipeline, 
                         anticipate_responses, get_responses, ensemble_responses, action_responses)

from .data_storage import DataStorage, DataCollection, SharedMemoryManager

from typing import List

import asyncio
import concurrent.futures
import os
import numpy as np



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

        # Store a reference to the DataStorage in the AutonoMachine.
        self.data_storage = in_data_storage

        # Keep a copy of the instructions that drive this ProblemSolver.
        self.instructions = in_instructions
        self.key_target = None
        self.keys_features = None

        # Note whether to use multiprocessing and how many processors are available.
        self.do_mp = do_mp
        self.n_procs = in_n_procs

        # Define semaphores to control concurrency during solution development. 
        self.semaphore_hpo = None
        self.semaphore_pipelines = None
        
        self.solution = None            # MLPipelines that are in production.
        self.queue_dev = None           # MLPipelines and HPO runs that are in development.
        # Note: This queue must not be instantiated as an asyncio queue until within a coroutine.
        self.hpo_runs_in_dev = dict()   # A dictionary to track what HPO runs are in development.
        self.pipelines_in_dev = dict()  # A dictionary to track what pipelines are in development.

        # Keep track of the data-storage instance up to which model has used.
        # TODO: Consider variant starting points for the model and update log messages.
        self.id_observations_last = 0
        self.id_queries_last = 0
        
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

        # Instantiate the solution as part of an event loop so that prerequisite data is ingested.
        self.solution = ProblemSolution(in_instructions = self.instructions, in_data_storage = self.data_storage)

        # Instantiate the development queue now that this code is running internally within an event loop.
        self.queue_dev = asyncio.Queue()

        strategy = self.instructions.strategy

        # Set up a semaphore to stop too many HPO runs operating concurrently.
        if strategy.max_hpo_concurrency is None:
            self.semaphore_hpo = asyncio.Semaphore(self.n_procs)
        else:
            self.semaphore_hpo = asyncio.Semaphore(strategy.max_hpo_concurrency)

        if strategy.do_defaults:
            for key_group in self.solution.groups:
                data_filter = self.solution.filters[key_group]
                pipelines = create_pipelines_default(in_keys_features = self.keys_features,
                                                     in_key_target = self.key_target,
                                                     in_strategy = strategy)
                dev_package = (pipelines, key_group, data_filter)
                await self.queue_dev.put(dev_package)

                # The queue counter will be decremented per pipeline or HPO run.
                # Thus ensure packaged pipelines are accompanied by an appropriate number of skippable tokens.
                for _ in range(len(pipelines)-1):
                    await self.queue_dev.put(None)
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running default pipelines.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        # TODO: Consider giving random pipelines some validation, not challenging with inf loss.
        if strategy.do_random:
            # TODO: Let user decide how many random pipelines to develop.
            for count_pipeline in range(1):
                for key_group in self.solution.groups:
                    data_filter = self.solution.filters[key_group]
                    pipeline = create_pipeline_random(in_keys_features = self.keys_features,
                                                      in_key_target = self.key_target,
                                                      in_strategy = strategy)
                    dev_package = (pipeline, key_group, data_filter)
                    await self.queue_dev.put(dev_package)
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running random pipelines.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        if strategy.do_hpo:
            if len(strategy.search_space.list_predictors()) > 0:
                for key_group in self.solution.groups:
                    data_filter = self.solution.filters[key_group]
                    dev_package = (HPOInstructions(in_strategy = strategy), key_group, data_filter)
                    await self.queue_dev.put(dev_package)
            else:
                text_warning = ("The Strategy for ProblemSolver '%s' does not suggest "
                                "any predictors in its search space.") % self.name
                log.warning("%s - %s" % (Timestamp(), text_warning))
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running HPO.") % self.name
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
        self.ops.append(create_async_task(self.process_development))
        self.ops.append(create_async_task(self.process_queries))
        # self.ops.extend([create_async_task(self.process_strategy),
        #                  create_async_task(self.process_queries)])
        group = asyncio.gather(*self.ops)
        try:
            await group
        except Exception as e:
            text_alert = ("%s - ProblemSolver '%s' encountered an error. "
                          "Cancelling Asyncio operations." % (Timestamp(), self.name))
            identify_exception(e, text_alert)
            for op in self.ops:
                op.cancel()

        self.is_running = False



    async def process_development(self):
        if self.do_mp:
            executor_class = concurrent.futures.ProcessPoolExecutor
            text_executor = "processes"
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
            text_executor = "threads"

        log.info("%s - ProblemSolver '%s' has launched a pool of %i %s "
                 "to train MLPipelines." % (Timestamp(), self.name, self.n_procs, text_executor))
        loop = asyncio.get_event_loop()

        with executor_class(max_workers = self.n_procs) as executor:
            while True:
                dev_package = await self.queue_dev.get()
                if dev_package is None:
                    continue
                object_dev = dev_package[0]
                key_group = dev_package[1]
                data_filter = dev_package[2]

                if key_group == "":
                    text_group = "."
                else:
                    text_group = ": " + key_group
                log.info("%s - Development request received for learner-group%s" 
                         % (Timestamp(), text_group))

                # # TODO: Decide what history of data to train pipelines on.
                # idx_start = 0
                # idx_stop = None
                # if idx_stop is None:
                #     idx_stop = observations.get_amount()

                info_process = {"keys_features": self.keys_features,
                                "key_target": self.key_target,
                                # "idx_start": idx_start,
                                # "idx_stop": idx_stop,
                                "n_available": self.data_storage.get_amount()}

                # Put solitary pipelines into a list.
                if isinstance(object_dev, MLPipeline):
                    object_dev = [object_dev]

                if isinstance(object_dev, list) and all(isinstance(item, MLPipeline) for item in object_dev):

                    create_async_task(self.process_pipelines, in_executor = executor, 
                                      in_pipelines = object_dev, in_info_process = info_process, 
                                      in_key_group = key_group, in_data_filter = data_filter)
                    
                elif isinstance(object_dev, HPOInstructions):

                    create_async_task(self.process_hpo, in_executor = executor, 
                                      in_hpo_instructions = object_dev, in_info_process = info_process, 
                                      in_key_group = key_group, in_data_filter = data_filter)

                # Let other asynchronous tasks proceed if they are waiting.
                # Prevents excessively unpacking development queue if development or inference is possible.
                await asyncio.sleep(0)

    async def process_pipelines(self, in_executor, in_pipelines: List[MLPipeline], in_info_process,
                                in_data_filter, in_key_group: str):

        loop = asyncio.get_event_loop()

        log.info("%s - Preparing development of MLPipelines: '%s'."
                % (Timestamp(), ", ".join([pipeline.name for pipeline in in_pipelines])))

        time_start = Timestamp().time
        dict_observations = self.data_storage.observations
        tag_to_collection_ids = self.data_storage.observations_tag_to_collection_ids
        observations = filter_observations(dict_observations, tag_to_collection_ids, in_data_filter)
        # observations = await loop.run_in_executor(in_executor, filter_observations, dict_observations, 
        #                                           tag_to_collection_ids, in_data_filter)
        time_end = Timestamp().time
        duration = time_end - time_start
        
        log.info("%s   Time taken to collate relevant data, applying filter as required: %.3f s"
                % (Timestamp(None), duration))

        time_start = Timestamp().time
        frac_validation = self.instructions.strategy.frac_validation
        observations, sets_training, sets_validation = prepare_data(observations, in_info_process, frac_validation)
        # observations, sets_training, sets_validation = await loop.run_in_executor(in_executor, prepare_data, 
        #                                                                           observations, in_info_process,
        #                                                                           frac_validation)
        time_end = Timestamp().time
        duration = time_end - time_start

        log.info("%s   Time taken to construct training (%0.2f) and validation (%0.2f) sets: %.3f s"
                % (Timestamp(None), 1 - frac_validation, frac_validation, duration))

        # Create a manager object that stores and loads data, possibly writing it efficiently to disk.
        time_start = Timestamp().time
        data_sharer = SharedMemoryManager(in_uses = len(in_pipelines), do_mp = self.do_mp)
        data_sharer.save_observations(in_observations = observations, 
                                        in_sets_training = sets_training, 
                                        in_sets_validation = sets_validation)
        time_end = Timestamp().time
        duration = time_end - time_start
        if self.do_mp:
            log.info("%s   Time taken to write observations and training/validation sets to temp files: %.3f s"
                    % (Timestamp(None), duration))

        # Develop the pipelines.
        for pipeline in in_pipelines:
            # future_pipeline = develop_pipeline(pipeline, observations, sets_training, sets_validation, in_info_process)
            future_pipeline = loop.run_in_executor(in_executor, develop_pipeline, pipeline, data_sharer, in_info_process)
            self.pipelines_in_dev[pipeline.name] = True
                
            create_async_task(self.push_to_production, future_pipeline, in_key_group = in_key_group, 
                              in_data_sharer = data_sharer)


    async def process_hpo(self, in_executor, in_hpo_instructions: HPOInstructions, in_info_process,
                          in_data_filter, in_key_group: str):
        
        async with self.semaphore_hpo:

            # Leave one processor free when running HPO for training specified pipelines.
            # TODO: Consider whether to block subsequent HPO requests if one is already running.
            n_procs_hpo = self.n_procs - 1

            loop = asyncio.get_event_loop()
            name_hpo = in_hpo_instructions.name

            # Ensure that active HPO runs have unique names.
            if name_hpo in self.hpo_runs_in_dev:
                text_error = ("HPOInstructions named '%s' was encountered in the development queue "
                              "while another identically named HPO run was active.") % name_hpo
                log.error("%s - %s" % (Timestamp(), text_error))
                raise Exception(text_error)
            
            log.info("%s - Preparing HPO run '%s'." % (Timestamp(), name_hpo))

            time_start = Timestamp().time
            dict_observations = self.data_storage.observations
            tag_to_collection_ids = self.data_storage.observations_tag_to_collection_ids
            observations = filter_observations(dict_observations, tag_to_collection_ids, in_data_filter)
            # observations = await loop.run_in_executor(in_executor, filter_observations, dict_observations,
            #                                           tag_to_collection_ids, in_data_filter)
            time_end = Timestamp().time
            duration = time_end - time_start
            
            log.info("%s   Time taken to collate relevant data, applying filter as required: %.3f s"
                     % (Timestamp(None), duration))

            sets_training = list()
            sets_validation = list()

            time_start = Timestamp().time
                
            frac_validation = in_hpo_instructions.frac_validation
            observations, sets_training, sets_validation = prepare_data(observations, in_info_process, frac_validation)
            # observations, sets_training, sets_validation = await loop.run_in_executor(in_executor, prepare_data,
            #                                                                           observations, in_info_process,
            #                                                                           frac_validation)

            time_end = Timestamp().time
            duration = time_end - time_start

            log.info("%s   Time taken to construct training (%0.2f) and validation (%0.2f) sets: %.3f s"
                     % (Timestamp(None), 1 - frac_validation, frac_validation, duration))

            # Create a manager object that stores and loads data, possibly writing it efficiently to disk.
            time_start = Timestamp().time
            data_sharer = SharedMemoryManager(do_mp = self.do_mp)
            data_sharer.save_observations(in_observations = observations, 
                                            in_sets_training = sets_training, 
                                            in_sets_validation = sets_validation)
            time_end = Timestamp().time
            duration = time_end - time_start
            if self.do_mp:
                log.info("%s   Time taken to write observations and training/validation sets to temp files: %.3f s"
                        % (Timestamp(None), duration))

            # Activate the HPO run.
            log.info("%s   Launching %i-worker HPO run '%s'." % (Timestamp(), n_procs_hpo, name_hpo))
            future_pipeline = loop.run_in_executor(in_executor, run_hpo, in_hpo_instructions, data_sharer, in_info_process)
            self.hpo_runs_in_dev[name_hpo] = True

            # Add HPO workers to the run.
            for idx_worker in range(1, n_procs_hpo):
                if name_hpo in self.hpo_runs_in_dev:
                    create_async_task(self.support_hpo, in_executor, idx_worker, in_hpo_instructions, data_sharer, in_info_process)
                
            # The await prevents the semaphore being released until an HPO ends in production.
            await create_async_task(self.push_to_production, future_pipeline,
                                    in_key_group = in_key_group, in_name_hpo = name_hpo, in_data_sharer = data_sharer)
        
        
    async def support_hpo(self, in_executor, in_idx_worker: int, in_hpo_instructions: HPOInstructions, 
                          in_data_sharer: SharedMemoryManager, in_info_process):
        
        loop = asyncio.get_event_loop()
        name_hpo = in_hpo_instructions.name

        add_attempts = 0
        while True:
            result = await loop.run_in_executor(in_executor, add_hpo_worker, in_hpo_instructions, in_data_sharer,
                                                in_info_process, in_idx_worker)
            
            if not isinstance(result, Exception):
                break
            else:
                log.warning("%s - Failed to add worker %i of HPO run '%s'. "
                            "Considering a reattempt." % (Timestamp(), in_idx_worker, name_hpo))
                # If the HPO run is no longer active, forget about it.
                if not name_hpo in self.hpo_runs_in_dev:
                    log.warning("%s   However, HPO run '%s' is no longer active." 
                                % (Timestamp(None), name_hpo))
                    break
                elif add_attempts > 5:
                    log.warning("%s   However, too many attempts have been made." 
                                % (Timestamp(None), name_hpo))
                    break
                # If the HPO run is active, the manager is delayed in starting up. Try again.
                await asyncio.sleep(1)
            add_attempts += 1



    async def push_to_production(self, in_future_pipeline, in_key_group: str, in_name_hpo: str = None, 
                                 in_data_sharer: SharedMemoryManager = None):
        try:
            pipeline, info_process = await in_future_pipeline

            if "text_hpo" in info_process:
                log.info(info_process["text_hpo"])

            # idx_start = info_process["idx_start"]
            # idx_stop = info_process["idx_stop"]
            duration_prep = info_process["duration_prep"]
            duration_proc = info_process["duration_proc"]
            n_instances = info_process["n_instances"]
            n_available = info_process["n_available"]
            y_last = pipeline.training_y_true[-1]
            y_pred_last = pipeline.training_y_response[-1]

            log.info("%s - Pipeline '%s' has learned from a total of %i observations (filtered from %i).\n"
                     "%s   Structure: %s\n"
                     "%s   Time taken to retrieve data: %.3f s\n"
                     "%s   Time taken to train/score pipeline on data: %.3f s\n"
                     "%s   Training loss: %f\n"
                     "%s   (Heuristic validation loss: %f)\n"
                     "%s   Last observation: Prediction '%s' vs True Value '%s'"
                     % (Timestamp(), pipeline.name, n_instances, n_available,
                        Timestamp(None), pipeline.components_as_string(do_hpars = True),
                        Timestamp(None), duration_prep,
                        Timestamp(None), duration_proc,
                        Timestamp(None), pipeline.get_loss(is_training = True),
                        Timestamp(None), pipeline.get_loss(),
                        Timestamp(None), y_pred_last, y_last))
            
            log.info("%s   Pipeline '%s' is pushed to production." % (Timestamp(None), pipeline.name))
            self.solution.insert_learner(in_pipeline = pipeline, in_key_group = in_key_group)
        except Exception as e:
            text_alert = "%s - ProblemSolver '%s' failed to process an MLPipeline." % (Timestamp(), self.name)
            identify_exception(e, text_alert)
        finally:
            # Mark the HPO run or pipeline as no longer in development.
            if not in_name_hpo is None:
                del self.hpo_runs_in_dev[in_name_hpo]
                log.info("%s   HPO runs still in development: %i" 
                         % (Timestamp(None), len(self.hpo_runs_in_dev)))
            else:
                if pipeline.name in self.pipelines_in_dev:
                    del self.pipelines_in_dev[pipeline.name]
                else:
                    log.warning("%s   Noted multiple attempts to clear pipeline '%s' from development. "
                                "Possibly a non-unique name." % (Timestamp(), pipeline.name))
                log.info("%s   Pipelines still in development: %i" 
                         % (Timestamp(None), len(self.pipelines_in_dev)))

            # Decrement the development queue.
            self.queue_dev.task_done()

            # Decrement the use counter on a shared memory manager, if it exists.
            # This may clean temporary files if no more pipelines are using the written data.
            if not in_data_sharer is None:
                in_data_sharer.decrement_uses()


    async def process_strategy(self):
        while True:

            # Check for new data and learn from it.
            if self.id_observations_last < self.data_storage.observations.get_amount():
                self.id_observations_last = self.data_storage.observations.get_amount()
                
            # TODO: Add things to the development queue as required.
                
            await self.data_storage.has_new_observations

    async def process_queries(self):

        # Do any preparatory work for responses, such as creating a folder for outputs.
        anticipate_responses()
        
        while True:

            # If required, ensure no pipelines are awaiting development before querying.
            if self.instructions.do_query_after_complete:
                await self.queue_dev.join()

            # Check if there are more instances in storage than have been processed.
            # If not, wait until new queries arive.
            if not self.id_queries_last < self.data_storage.id_data_last:
                await self.data_storage.has_new_queries

            # Fix how many queries to process based on what is available at the time.
            id_stop = self.data_storage.id_data_last

            info_process = {"keys_features": self.keys_features,
                            "key_target": self.key_target,
                            "id_start": self.id_queries_last,
                            "id_stop": id_stop}

            for collection_id, queries in self.data_storage.queries.items():
                queries, _ = queries.split_by_special_range(in_id_start_exclusive = self.id_queries_last,
                                                            in_id_stop_inclusive = id_stop)
                
                queries, _, _ = prepare_data(in_collection = queries, 
                                             in_info_process = info_process,
                                             in_n_sets = 0)
                
                responses_dict = dict()
                
                for tags, list_pipelines in self.solution.groups.items():

                    responses_dict[tags] = dict()

                    for rank_pipeline, pipeline in enumerate(list_pipelines):

                        responses_dict[tags][rank_pipeline] = dict()

                        responses, pipeline, info_process = get_responses(in_pipeline = pipeline,
                                                                          in_queries = queries,
                                                                          in_info_process = info_process)
                        # print(pipeline)
                        # print(responses)

                        responses_dict[tags][rank_pipeline]["responses"] = responses
                        record_loss = [None]*queries.get_amount()
                        record_loss[-1] = pipeline.get_loss()
                        responses_dict[tags][rank_pipeline]["loss"] = record_loss
                        responses_dict[tags][rank_pipeline]["name"] = [pipeline.name]*queries.get_amount()

                responses_best = ensemble_responses(responses_dict)

                tag_queries = self.data_storage.get_tag_combo_from_collection_id(collection_id,
                                                                                 as_string = True,
                                                                                 as_query = True)
                action_responses(in_queries = queries,
                                 in_responses_best = responses_best,
                                 in_responses_dict = responses_dict,
                                 in_collection_tag_string = tag_queries,
                                 in_keys_features = self.keys_features,
                                 in_key_target = self.key_target,
                                 in_solution = self.solution)

            #         log.info("%s - Pipeline '%s' has responded to a total of %i queries.\n"
            #                 "%s   Structure: %s\n"
            #                 "%s   Time taken to retrieve data: %.3f s\n"
            #                 "%s   Time taken to query/score pipeline on data: %.3f s\n"
            #                 "%s   (Training loss: %f)\n"
            #                 "%s   Testing loss: %f\n"
            #                 "%s   Last observation: Prediction '%s' vs True Value '%s'"
            #                 % (Timestamp(), pipeline.name, idx_stop - idx_start,
            #                     Timestamp(None), pipeline.components_as_string(do_hpars = True),
            #                     Timestamp(None), duration_prep,
            #                     Timestamp(None), duration_proc,
            #                     Timestamp(None), pipeline.get_loss(is_training = True),
            #                     Timestamp(None), pipeline.get_loss(),
            #                     Timestamp(None), y_pred_last, y_last))
                
            # Update an index to acknowledge the queries that have been processed.
            self.id_queries_last = id_stop

            # Let other asynchronous tasks proceed if they are waiting.
            # Prevents endlessly focussing on inference if development is possible.
            await asyncio.sleep(0)
            
    # TODO: Include error checking for no features. Error-check target existence somewhere too.
    async def set_target_and_features(self, in_key_target: str,
                                      in_keys_features: List[str] = None, 
                                      do_exclude: bool = False):
        
        storage_keys = self.data_storage.get_keys()

        if in_key_target in storage_keys:
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
                if key_feature in storage_keys:
                    keys_features.append(key_feature)
                else:
                    log.warning("%s - Desired feature key '%s' cannot be found in DataStorage.\n"
                                "%s   The ProblemSolver will ignore it." 
                                % (Timestamp(), key_feature, Timestamp(None)))
        # Otherwise, include every feature existing in the data storage...
        # Except for feature keys specified with the intention of excluding.
        else:
            for dkey in storage_keys:
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
        for tags, list_pipelines in self.solution.groups.items():
            for pipeline in list_pipelines:
                figs.extend(pipeline.inspect_structure())
                figs.extend(pipeline.inspect_performance(for_training = True))
                figs.extend(pipeline.inspect_performance(for_training = False))
        return figs