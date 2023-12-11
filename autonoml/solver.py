# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .settings import SystemSettings as SS
from .utils import log, Timestamp, identify_exception
from .concurrency import create_async_task_from_sync, create_async_task
from .pipeline import MLPipeline
from .hpo import HPOInstructions, run_hpo, add_hpo_worker, create_pipelines_default, create_pipeline_random
from .solution import ProblemSolverInstructions, ProblemSolution
from .strategy import Strategy
from .solver_ops import (ProcessInformation,
                         filter_observations, prepare_data, develop_pipeline,
                         adapt_to_data, track_dynamics,
                         get_responses, ensemble_responses, action_responses)

from .data_storage import DataStorage, SharedMemoryManager

from typing import List

import asyncio
import concurrent.futures




class ProblemSolver:
    """
    A manager for pipelines and components that learn from data and respond to queries.
    """

    count = 0
    
    def __init__(self, in_data_storage: DataStorage, 
                 in_instructions: ProblemSolverInstructions,
                 in_strategy: Strategy = None,
                 in_n_procs: int = 1, do_mp: bool = False):
        
        self.name = "Sol_" + str(ProblemSolver.count)
        ProblemSolver.count += 1
        log.info("%s - Initialising ProblemSolver '%s'." % (Timestamp(), self.name))

        # Store a reference to the DataStorage in the AutonoMachine.
        self.data_storage = in_data_storage

        # Keep a reference to the instructions that drive this ProblemSolver.
        # Include the feature/target keys that will be inferred later from the instructions.
        self.instructions = in_instructions
        self.key_target = None
        self.keys_features = None

        # Create a default strategy if user did not provide one.
        self.strategy = Strategy() if in_strategy is None else in_strategy

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
        self.id_observations_last = -1
        self.id_queries_last = -1
        
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
        self.solution = ProblemSolution(in_instructions = self.instructions, in_strategy = self.strategy, 
                                        in_data_storage = self.data_storage)

        # Instantiate the development queue now that this code is running internally within an event loop.
        self.queue_dev = asyncio.Queue()

        # Set up a semaphore to stop too many HPO runs operating concurrently.
        if self.strategy.max_hpo_concurrency is None:
            self.semaphore_hpo = asyncio.Semaphore(self.n_procs)
        else:
            self.semaphore_hpo = asyncio.Semaphore(self.strategy.max_hpo_concurrency)

        # Prepare to develop pipelines on all the data is presently available.
        self.id_observations_last = self.data_storage.id_data_last
        info_process = ProcessInformation(in_keys_features = self.keys_features,
                                          in_key_target = self.key_target,
                                          in_id_last_new = self.data_storage.id_data_last)
        info_process.set_n_available(self.data_storage.get_amount())

        attempt = 0
        if self.strategy.do_defaults:
            for key_group in self.solution.groups:
                data_filter = self.solution.filters[key_group]
                pipelines = create_pipelines_default(in_keys_features = self.keys_features,
                                                     in_key_target = self.key_target,
                                                     in_strategy = self.strategy)
                dev_package = (pipelines, key_group, data_filter, info_process, attempt)
                await self.queue_dev.put(dev_package)

                # The queue counter will be decremented per pipeline or HPO run.
                # Thus ensure packaged pipelines are accompanied by an appropriate number of skippable tokens.
                for _ in range(len(pipelines)-1):
                    await self.queue_dev.put(None)
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running default pipelines.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        if self.strategy.do_random:
            for _ in range(self.strategy.n_samples):
                for key_group in self.solution.groups:
                    data_filter = self.solution.filters[key_group]
                    pipeline = create_pipeline_random(in_keys_features = self.keys_features,
                                                      in_key_target = self.key_target,
                                                      in_strategy = self.strategy)
                    dev_package = (pipeline, key_group, data_filter, info_process, attempt)
                    await self.queue_dev.put(dev_package)
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running random pipelines.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        if self.strategy.do_hpo:
            if len(self.strategy.search_space.list_predictors()) > 0:
                for key_group in self.solution.groups:
                    data_filter = self.solution.filters[key_group]
                    dev_package = (HPOInstructions(in_strategy = self.strategy), 
                                   key_group, data_filter, info_process, attempt)
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
        self.ops.append(create_async_task(self.process_observations))
        self.ops.append(create_async_task(self.process_queries))
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

                if key_group == "":
                    text_group = "."
                else:
                    text_group = ": " + key_group
                log.info("%s - Development request received for learner-group%s" 
                         % (Timestamp(), text_group))

                # Put solitary pipelines into a list.
                if isinstance(object_dev, MLPipeline):
                    object_dev = [object_dev]
                    dev_package = (object_dev,) + dev_package[1:]

                if isinstance(object_dev, list) and all(isinstance(item, MLPipeline) for item in object_dev):

                    create_async_task(self.process_pipelines, in_executor = executor, 
                                      in_dev_package = dev_package)
                    
                elif isinstance(object_dev, HPOInstructions):

                    create_async_task(self.process_hpo, in_executor = executor, 
                                      in_dev_package = dev_package)

                # Let other asynchronous tasks proceed if they are waiting.
                # Prevents excessively unpacking development queue if development or inference is possible.
                await asyncio.sleep(0)

    async def process_pipelines(self, in_executor, in_dev_package):

        pipelines = in_dev_package[0]
        data_filter = in_dev_package[2]
        info_process = in_dev_package[3]

        loop = asyncio.get_event_loop()

        log.info("%s - Preparing development of MLPipelines: '%s'."
                % (Timestamp(), ", ".join([pipeline.name for pipeline in pipelines])))

        time_start = Timestamp().time
        dict_observations = self.data_storage.observations
        tag_to_collection_ids = self.data_storage.observations_tag_to_collection_ids
        observations = filter_observations(dict_observations, tag_to_collection_ids, data_filter)
        # observations = await loop.run_in_executor(in_executor, filter_observations, dict_observations, 
        #                                           tag_to_collection_ids, data_filter)
        time_end = Timestamp().time
        duration = time_end - time_start
        
        log.info("%s   Time taken to collate relevant data, applying filter as required: %.3f s"
                % (Timestamp(None), duration))

        time_start = Timestamp().time
        frac_validation = self.strategy.frac_validation
        folds_validation = self.strategy.folds_validation
        observations, sets_training, sets_validation = prepare_data(observations, info_process, 
                                                                    frac_validation, folds_validation)
        # observations, sets_training, sets_validation = await loop.run_in_executor(in_executor, prepare_data, 
        #                                                                           observations, info_process,
        #                                                                           frac_validation)
        time_end = Timestamp().time
        duration = time_end - time_start

        log.info("%s   Time taken to construct training (%0.2f) and validation (%0.2f) sets: %.3f s"
                % (Timestamp(None), 1 - frac_validation, frac_validation, duration))

        # Create a manager object that stores and loads data, possibly writing it efficiently to disk.
        time_start = Timestamp().time
        data_sharer = SharedMemoryManager(in_uses = len(pipelines), do_mp = self.do_mp)
        data_sharer.save_observations(in_observations = observations, 
                                        in_sets_training = sets_training, 
                                        in_sets_validation = sets_validation)
        time_end = Timestamp().time
        duration = time_end - time_start
        if self.do_mp:
            log.info("%s   Time taken to write observations and training/validation sets to temp files: %.3f s"
                    % (Timestamp(None), duration))

        # Develop the pipelines.
        for pipeline in pipelines:
            # future_pipeline = develop_pipeline(pipeline, observations, sets_training, sets_validation, info_process)
            future_pipeline = loop.run_in_executor(in_executor, develop_pipeline, pipeline, data_sharer, info_process)
            self.pipelines_in_dev[pipeline.name] = True
                
            create_async_task(self.push_to_production, future_pipeline, 
                              in_dev_package = (pipeline,) + in_dev_package[1:], 
                              in_data_sharer = data_sharer)


    async def process_hpo(self, in_executor, in_dev_package):
        
        async with self.semaphore_hpo:

            hpo_instructions = in_dev_package[0]
            data_filter = in_dev_package[2]
            info_process = in_dev_package[3]

            # Leave one processor free when running HPO for training specified pipelines.
            # TODO: Consider whether to block subsequent HPO requests if one is already running.
            n_procs_hpo = self.n_procs - 1

            loop = asyncio.get_event_loop()
            name_hpo = hpo_instructions.name

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
            observations = filter_observations(dict_observations, tag_to_collection_ids, data_filter)
            # observations = await loop.run_in_executor(in_executor, filter_observations, dict_observations,
            #                                           tag_to_collection_ids, data_filter)
            time_end = Timestamp().time
            duration = time_end - time_start
            
            log.info("%s   Time taken to collate relevant data, applying filter as required: %.3f s"
                     % (Timestamp(None), duration))

            sets_training = list()
            sets_validation = list()

            time_start = Timestamp().time
                
            frac_validation = hpo_instructions.frac_validation
            folds_validation = hpo_instructions.folds_validation
            observations, sets_training, sets_validation = prepare_data(observations, info_process, 
                                                                        frac_validation, folds_validation)
            # observations, sets_training, sets_validation = await loop.run_in_executor(in_executor, prepare_data,
            #                                                                           observations, info_process,
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
            future_pipeline = loop.run_in_executor(in_executor, run_hpo, hpo_instructions, data_sharer, info_process)
            self.hpo_runs_in_dev[name_hpo] = True

            # Add HPO workers to the run.
            for idx_worker in range(1, n_procs_hpo):
                if name_hpo in self.hpo_runs_in_dev:
                    create_async_task(self.support_hpo, in_executor, idx_worker, hpo_instructions, data_sharer, info_process)
                
            # The await prevents the semaphore being released until an HPO ends in production.
            await create_async_task(self.push_to_production, future_pipeline,
                                    in_dev_package = in_dev_package, 
                                    in_data_sharer = data_sharer)
        
        
    async def support_hpo(self, in_executor, in_idx_worker: int, in_hpo_instructions: HPOInstructions, 
                          in_data_sharer: SharedMemoryManager, in_info_process: ProcessInformation):
        
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



    async def push_to_production(self, in_future_pipeline, in_dev_package,
                                 in_data_sharer: SharedMemoryManager = None):
        
        object_dev = in_dev_package[0]
        key_group = in_dev_package[1]
        data_filter = in_dev_package[2]
        info_process_old = in_dev_package[3]
        attempt = in_dev_package[4]

        try:
            pipeline, info_process = await in_future_pipeline

            if not info_process.text is None:
                log.info(info_process.text)

            duration_prep = info_process.duration_prep
            duration_proc = info_process.duration_proc
            n_instances = info_process.n_instances
            n_available = info_process.n_available
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
            self.solution.insert_learner(in_pipeline = pipeline, in_key_group = key_group)

        except Exception as e:
            # Create an alert for the exception.
            if isinstance(object_dev, HPOInstructions):
                text_object = "HPO run '%s'" % object_dev.name
            # elif isinstance(object_dev, list) and all(isinstance(item, MLPipeline) for item in object_dev):
            #     text_object = "MLPipeline '%s'" % object_dev[0].name
            elif isinstance(object_dev, MLPipeline):
                text_object = "MLPipeline '%s'" % object_dev.name
            text_alert = "%s - ProblemSolver '%s' failed to process %s." % (Timestamp(), self.name, text_object)
            identify_exception(e, text_alert)

            # Retry development if appropriate.
            if attempt < SS.MAX_ATTEMPTS_DEVELOPMENT:
                time_to_wait = SS.BASE_DELAY_BEFORE_RETRY * (2**attempt)
                log.info("%s   Delay before reinsertion into the development queue: %s s" 
                         % (Timestamp(None), time_to_wait))
                await asyncio.sleep(time_to_wait)

                # Update the data range in case that was the problem.
                info_process_old.id_data_last = self.data_storage.id_data_last
                info_process_old.set_n_available(self.data_storage.get_amount())
                dev_package = (object_dev, key_group, data_filter, info_process_old, attempt + 1)
                await self.queue_dev.put(dev_package)
            else:
                log.warning("%s   Abandoning any further attempt on %s." % (Timestamp(None), text_object))

        finally:
            # Mark the HPO run or pipeline as no longer in development.
            if isinstance(object_dev, HPOInstructions):
                del self.hpo_runs_in_dev[object_dev.name]
                log.info("%s   HPO runs still in development: %i" 
                         % (Timestamp(None), len(self.hpo_runs_in_dev)))
            # elif isinstance(object_dev, list) and all(isinstance(item, MLPipeline) for item in object_dev):
            elif isinstance(object_dev, MLPipeline):
                if object_dev.name in self.pipelines_in_dev:
                    del self.pipelines_in_dev[object_dev.name]
                else:
                    log.warning("%s   Noted multiple attempts to clear pipeline '%s' from development. "
                                "Possibly a non-unique name." % (Timestamp(), object_dev.name))
                log.info("%s   Pipelines still in development: %i" 
                         % (Timestamp(None), len(self.pipelines_in_dev)))

            # Decrement the development queue.
            self.queue_dev.task_done()

            # Decrement the use counter on a shared memory manager, if it exists.
            # This may clean temporary files if no more pipelines are using the written data.
            if not in_data_sharer is None:
                in_data_sharer.decrement_uses()

    # TODO: This slows over time, possibly due to data filtering. Investigate/upgrade for efficiency.
    async def process_observations(self):
        
        while True:

            # Check if there are more instances in storage than have been processed.
            # If not, wait until new observations arive.
            if not self.id_observations_last < self.data_storage.id_data_last:
                await self.data_storage.has_new_observations

            # Fix how many observations to process based on what is available at the time.
            id_stop = self.data_storage.id_data_last

            info_process = ProcessInformation(in_keys_features = self.keys_features,
                                              in_key_target = self.key_target,
                                              in_id_last_old = self.id_observations_last,
                                              in_id_last_new = id_stop)
            
            # Go through each learner group and grab the data the learners would be interested in.
            # This involves filtering out irrelevant data according to tags and grabbing the right range.
            # This also involves preparing the data in X and Y format, i.e. feature/target space.
            for key_group in self.solution.groups:
                data_filter = self.solution.filters[key_group]

                time_start = Timestamp().time
                dict_observations = self.data_storage.observations
                tag_to_collection_ids = self.data_storage.observations_tag_to_collection_ids
                observations = filter_observations(dict_observations, tag_to_collection_ids, data_filter,
                                                   in_info_process = info_process)
                # print(observations.data)
                observations, _, _ = prepare_data(in_collection = observations,
                                                  in_info_process = info_process,
                                                  in_n_sets = 0)
                # print(observations.x)
                time_end = Timestamp().time
                duration = time_end - time_start
                
                # print(data_filter)
                # print(duration)
                # print(observations.get_amount())
                # print(observations.x.num_rows)
                # print(len(observations.ids))
                # print(len(observations.timestamps))

                # No point adapting if the new data has not landed in a particular allocation.
                if observations.get_amount() > 0:

                    results_dict = dict()
                    
                    # Pop learners from the group, taking note of current rank, and adapt them to the new data.
                    rank_pipeline = 0
                    list_adapted = list()
                    while len(self.solution.groups[key_group]) > 0:
                        pipeline = self.solution.groups[key_group].pop(0)

                        results_dict[rank_pipeline] = dict()

                        responses, pipeline, info_process = adapt_to_data(in_pipeline = pipeline,
                                                                        in_observations = observations,
                                                                        in_info_process = info_process)

                        list_adapted.append(pipeline)

                        results_dict[rank_pipeline]["responses"] = responses
                        record_loss = [None]*observations.get_amount()
                        record_loss[-1] = pipeline.get_loss()
                        results_dict[rank_pipeline]["loss"] = record_loss
                        results_dict[rank_pipeline]["name"] = [pipeline.name]*observations.get_amount()

                        rank_pipeline += 1

                    # Reinsert learners into their appropriate groups, which will re-sort them based on new losses.
                    while len(list_adapted) > 0:
                        self.solution.insert_learner(in_pipeline = list_adapted.pop(0), in_key_group = key_group)

                    track_dynamics(in_observations = observations,
                                   in_results_dict = results_dict,
                                   in_key_group = key_group,
                                   in_solution = self.solution,
                                   in_info_process = info_process)
                
            # Update an index to acknowledge the observations that have been processed.
            self.id_observations_last = id_stop

            # Let other asynchronous tasks proceed if they are waiting.
            # Prevents endlessly focussing on adaptation if inference is possible.
            await asyncio.sleep(0)

    async def process_queries(self):
        
        while True:

            # If required, ensure no pipelines are awaiting development before querying.
            if not self.instructions.do_immediate_responses:
                await self.queue_dev.join()

            # Check if there are more instances in storage than have been processed.
            # If not, wait until new queries arive.
            if not self.id_queries_last < self.data_storage.id_data_last:
                await self.data_storage.has_new_queries

            # Fix how many queries to process based on what is available at the time.
            id_stop = self.data_storage.id_data_last

            info_process = ProcessInformation(in_keys_features = self.keys_features,
                                              in_key_target = self.key_target,
                                              in_id_last_old = self.id_queries_last,
                                              in_id_last_new = id_stop)

            for collection_id, queries in self.data_storage.queries.items():
                queries, _ = queries.split_by_special_range(in_id_start_exclusive = self.id_queries_last,
                                                            in_id_stop_inclusive = id_stop)
                
                queries, _, _ = prepare_data(in_collection = queries, 
                                             in_info_process = info_process,
                                             in_n_sets = 0)
                
                results_dict = dict()
                
                for tags, list_pipelines in self.solution.groups.items():

                    results_dict[tags] = dict()

                    for rank_pipeline, pipeline in enumerate(list_pipelines):

                        results_dict[tags][rank_pipeline] = dict()

                        responses, pipeline, info_process = get_responses(in_pipeline = pipeline,
                                                                          in_queries = queries,
                                                                          in_info_process = info_process)

                        results_dict[tags][rank_pipeline]["responses"] = responses
                        record_loss = [None]*queries.get_amount()
                        record_loss[-1] = pipeline.get_loss()
                        results_dict[tags][rank_pipeline]["loss"] = record_loss
                        results_dict[tags][rank_pipeline]["name"] = [pipeline.name]*queries.get_amount()

                responses_best = ensemble_responses(results_dict)

                tag_queries = self.data_storage.get_tag_combo_from_collection_id(collection_id,
                                                                                 as_string = True,
                                                                                 as_query = True)

                action_responses(in_queries = queries,
                                 in_responses_best = responses_best,
                                 in_results_dict = results_dict,
                                 in_collection_tag_string = tag_queries,
                                 in_solution = self.solution,
                                 in_info_process = info_process)
                
            # Update an index to acknowledge the queries that have been processed.
            self.id_queries_last = id_stop

            # Let other asynchronous tasks proceed if they are waiting.
            # Prevents endlessly focussing on inference if development is possible.
            await asyncio.sleep(0)


            
    # TODO: Include error checking for no features. Error-check target existence somewhere too.
    async def set_target_and_features(self, in_key_target: str,
                                      in_keys_features: List[str] = None, 
                                      do_exclude: bool = False):
        
        storage_keys = self.data_storage.get_key_dict()

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