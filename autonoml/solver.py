# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:58:33 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp, identify_error
from .concurrency import create_async_task_from_sync, create_async_task
from .pipeline import MLPipeline, train_pipeline, test_pipeline
from .hpo import HPOInstructions, run_hpo, add_hpo_worker
from .strategy import Strategy

from .data_storage import DataStorage

# import __main__
import asyncio
import concurrent.futures
# import dill
# import multiprocess as mp
# from copy import deepcopy
import time



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

        self.n_challengers = 2

class ProblemSolution:
    """
    A container for all learners currently in production.
    """
    def __init__(self, in_instructions: ProblemSolverInstructions):
        self.learners = dict()
        self.learners[0] = list()

        self.n_challengers = in_instructions.n_challengers

    def insert_learner(self, in_pipeline: MLPipeline, in_tags = None):
        list_pipelines = self.learners[0]
        list_pipelines.append(in_pipeline)
        self.learners[0] = sorted(list_pipelines, key=lambda p: p.get_loss())
        log.debug(["%s: %0.2f" % (pipeline.name, pipeline.get_loss()) for pipeline in self.learners[0]])
        if len(self.learners[0]) > self.n_challengers + 1:
            pipeline_removed = self.learners[0].pop()
            log.debug("Removing uncompetitive challenger pipeline '%s' with loss: %0.2f" 
                      % (pipeline_removed.name, pipeline_removed.get_loss()))


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
        
        self.solution = ProblemSolution(in_instructions = in_instructions)
        # self.solution = dict()         # MLPipelines that are in production.
        self.queue_dev = None           # MLPipelines and HPO runs that are in development.
        # Note: This queue must not be instantiated as an asyncio queue until within a coroutine.
        self.hpo_runs_active = dict()   # A dictionary to track what HPO runs are active.

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
                await self.queue_dev.put(HPOInstructions(in_strategy = strategy))
                await self.queue_dev.put(HPOInstructions(in_strategy = strategy))
                await self.queue_dev.put(HPOInstructions(in_strategy = strategy))
            else:
                text_warning = ("The Strategy for ProblemSolver '%s' does not suggest "
                                "any predictors in its search space.") % self.name
                log.warning("%s - %s" % (Timestamp(), text_warning))
        else:
            text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
                         "running HPO.") % self.name
            log.info("%s - %s" % (Timestamp(), text_info))

        # # TODO: Do something genuine with custom pipelines.
        # if strategy.do_custom:
        #     for count_pipeline in range(1):
        #         await self.add_pipeline(create_pipeline_random(in_keys_features = self.keys_features,
        #                                                        in_key_target = self.key_target))
        # else:
        #     text_info = ("The Strategy for ProblemSolver '%s' does not suggest "
        #                  "running custom pipelines.") % self.name
        #     log.info("%s - %s" % (Timestamp(), text_info))

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
                idx_stop = None
                if idx_stop is None:
                    idx_stop = self.data_storage.observations.get_amount()

                name_hpo = None
                info_process = {"keys_features": self.keys_features,
                                "key_target": self.key_target,
                                "idx_start": idx_start,
                                "idx_stop": idx_stop}
                
                observations, _ = self.data_storage.observations.split_by_range(in_idx_start = idx_start,
                                                                                in_idx_stop = idx_stop)

                if isinstance(object_dev, MLPipeline):
                    future_pipeline = loop.run_in_executor(executor, train_pipeline, object_dev,
                                                           observations, info_process)
                    
                elif isinstance(object_dev, HPOInstructions):

                    name_hpo = object_dev.name

                    # Ensure that active HPO runs have unique names.
                    if name_hpo in self.hpo_runs_active:
                        text_error = ("HPOInstructions named '%s' was encountered in the development queue "
                                      "while another identically named HPO run was active.") % name_hpo
                        log.error("%s - %s" % (Timestamp(), text_error))
                        raise Exception(text_error)

                    log.info("%s - Preparing HPO run '%s'." % (Timestamp(), name_hpo))
                    sets_training = list()
                    sets_validation = list()

                    time_start = Timestamp().time
                    # TODO: Let users decide how many training/validation pairs to form.
                    for idx_set in range(1):
                        set_validation, set_training = observations.split_by_fraction(in_fraction = object_dev.frac_validation)
                        sets_training.append(set_training)
                        sets_validation.append(set_validation)
                    time_end = Timestamp().time
                    duration = time_end - time_start

                    log.info("%s   Time taken to construct training (%0.2f) and validation (%0.2f) sets: %.3f s" 
                             % (Timestamp(None), 1 - object_dev.frac_validation, object_dev.frac_validation, duration))

                    # Activate the HPO run.
                    log.info("%s - Launching %i-worker HPO run '%s'." 
                             % (Timestamp(), n_procs_hpo, name_hpo))
                    future_pipeline = loop.run_in_executor(executor, run_hpo, object_dev, observations,
                                                           sets_training, sets_validation, info_process)
                    self.hpo_runs_active[name_hpo] = True

                    # Add HPO workers to the run.
                    for idx_worker in range(1, n_procs_hpo):
                        # if name_hpo in self.hpo_runs_active:
                        create_async_task(self.support_hpo, executor, idx_worker, object_dev,
                                          sets_training, sets_validation, info_process)
                        
                create_async_task(self.push_to_production, future_pipeline, name_hpo)

    async def support_hpo(self, in_executor, in_idx_worker: int, in_hpo_instructions, 
                          in_sets_training, in_sets_validation, in_info_process):
        
        loop = asyncio.get_event_loop()
        name_hpo = in_hpo_instructions.name

        add_attempts = 0
        while True:
            result = await loop.run_in_executor(in_executor, add_hpo_worker, in_hpo_instructions, 
                                    in_sets_training, in_sets_validation, in_info_process, in_idx_worker)
            
            if not isinstance(result, Exception):
                break
            else:
                log.warning("%s - Failed to add worker %i of HPO run '%s'. "
                            "Considering a reattempt." % (Timestamp(), in_idx_worker, name_hpo))
                # If the HPO run is no longer active, forget about it.
                if not name_hpo in self.hpo_runs_active:
                    log.warning("%s   However, HPO run '%s' is no longer active." 
                                % (Timestamp(None), name_hpo))
                    break
                elif add_attempts > 5:
                    log.warning("%s   However, too many attempts have been made." 
                                % (Timestamp(None), name_hpo))
                    break
                # If the HPO run is active, the manager is delayed in starting up. Try again.
                time.sleep(1)
            add_attempts += 1


    async def push_to_production(self, in_future_pipeline, in_name_hpo: str = None):
        try:
            pipeline, info_process = await in_future_pipeline

            if "text_hpo" in info_process:
                log.info(info_process["text_hpo"])

            idx_start = info_process["idx_start"]
            idx_stop = info_process["idx_stop"]
            duration_prep = info_process["duration_prep"]
            duration_proc = info_process["duration_proc"]
            y_last = pipeline.training_y_true[-1]
            y_pred_last = pipeline.training_y_response[-1]

            log.info("%s - Pipeline '%s' has learned from a total of %i observations.\n"
                     "%s   Structure: %s\n"
                     "%s   Time taken to retrieve data: %.3f s\n"
                     "%s   Time taken to train/score pipeline on data: %.3f s\n"
                     "%s   Training loss: %f\n"
                     "%s   (Testing loss: %f)\n"
                     "%s   Last observation: Prediction '%s' vs True Value '%s'"
                     % (Timestamp(), pipeline.name, idx_stop - idx_start,
                        Timestamp(None), pipeline.components_as_string(do_hpars = True),
                        Timestamp(None), duration_prep,
                        Timestamp(None), duration_proc,
                        Timestamp(None), pipeline.get_loss(is_training = True),
                        Timestamp(None), pipeline.get_loss(),
                        Timestamp(None), y_pred_last, y_last))
            
            self.solution.insert_learner(pipeline)
            log.info("%s   Pipeline '%s' is pushed to production." % (Timestamp(None), pipeline.name))
        except Exception as e:
            text_alert = "%s - ProblemSolver '%s' failed to process an MLPipeline." % (Timestamp(), self.name)
            identify_error(e, text_alert)
        finally:
            # Mark the HPO run as inactive, if applicable.
            if not in_name_hpo is None:
                del self.hpo_runs_active[in_name_hpo]

            self.queue_dev.task_done()

    async def process_strategy(self):
        while True:

            # Check for new data and learn from it.
            if self.idx_data < self.data_storage.observations.get_amount():
                self.idx_data = self.data_storage.observations.get_amount()
                
            # TODO: Add things to the development queue as required.
                
            await self.data_storage.has_new_observations

    async def process_queries(self):
        
        while True:

            # If required, ensure no pipelines are awaiting development before querying.
            if self.instructions.do_query_after_complete:
                await self.queue_dev.join()

            # Check if there are more queries in storage than have been processed.
            # If not, wait until new queries arive.
            # TODO: Compare against a data storage index rather than a length.
            if not self.idx_queries < self.data_storage.queries.get_amount():
                await self.data_storage.has_new_queries

            # Fix how many queries to process based on what is available at the time.
            idx_stop = self.data_storage.queries.get_amount()

            info_process = {"keys_features": self.keys_features,
                            "key_target": self.key_target,
                            "idx_start": self.idx_queries,
                            "idx_stop": idx_stop}

            # Grab the newest queries as a data collection.
            queries, _ = self.data_storage.queries.split_by_range(in_idx_start = self.idx_queries,
                                                                  in_idx_stop = idx_stop)

            # Go through every pipeline in production and process the queries.
            # TODO: Ensemble them to derive a single set of responses.
            for tags, list_pipelines in self.solution.learners.items():
                # try:
                #     pipeline = self.solution[pipeline_key]
                # except Exception as e:
                #     log.warning("%s - Pipeline '%s' disappeared in the middle of a query phase. "
                #                 "Continuing to query the next pipeline." % (Timestamp(), pipeline_key))
                #     log.debug(e)
                #     continue

                for pipeline in list_pipelines:

                    pipeline, metric = test_pipeline(in_pipeline = pipeline, in_data_collection = queries, 
                                                    in_info_process = info_process)

                    idx_start = info_process["idx_start"]
                    idx_stop = info_process["idx_stop"]
                    duration_prep = info_process["duration_prep"]
                    duration_proc = info_process["duration_proc"]
                    # metric = info_process["metric"]
                    y_last = pipeline.testing_y_true[-1]
                    y_pred_last = pipeline.testing_y_response[-1]

                    log.info("%s - Pipeline '%s' has responded to a total of %i queries.\n"
                            "%s   Structure: %s\n"
                            "%s   Time taken to retrieve data: %.3f s\n"
                            "%s   Time taken to query/score pipeline on data: %.3f s\n"
                            "%s   (Training loss: %f)\n"
                            "%s   Testing loss: %f\n"
                            "%s   Last observation: Prediction '%s' vs True Value '%s'"
                            % (Timestamp(), pipeline.name, idx_stop - idx_start,
                                Timestamp(None), pipeline.components_as_string(do_hpars = True),
                                Timestamp(None), duration_prep,
                                Timestamp(None), duration_proc,
                                Timestamp(None), pipeline.get_loss(is_training = True),
                                Timestamp(None), pipeline.get_loss(),
                                Timestamp(None), y_pred_last, y_last))
                
            # Update an index to acknowledge the queries that have been processed.
            self.idx_queries = idx_stop

            # Let other asynchronous tasks proceed if they are waiting.
            # Prevents endlessly focussing on inference if development is possible.
            await asyncio.sleep(0)
            
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
        for tags, list_pipelines in self.solution.learners.items():
            for pipeline in list_pipelines:
                figs.extend(pipeline.inspect_structure())
                figs.extend(pipeline.inspect_performance(for_training = True))
                figs.extend(pipeline.inspect_performance(for_training = False))
        return figs