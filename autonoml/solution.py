# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:23 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pipeline import MLPipeline
from .strategy import Strategy

from .data_storage import DataStorage

from typing import Dict, List, Tuple, Union

import joblib

from enum import Enum
import os
import shutil

# TODO: Make these options clear to a user.
class AllocationMethod(Enum):
    """
    Descriptors for how to subdivide data per learner.
    """
    ONE_EACH = 0
    LEAVE_ONE_OUT = 1

# TODO: Let a user specify more details about a problem to constrict the strategy automatically.
class ProblemSolverInstructions:
    """
    Intended as a description of an ML problem to solve.
    Contrasts with a Strategy, which describes how to approach the problem.
    """
    def __init__(self, in_key_target: str, in_keys_features = None, do_exclude: bool = False,
                 do_immediate_responses: bool = True,
                 in_tags_allocation: List[Union[str, Tuple[str, AllocationMethod]]] = None,
                 in_directory_import: str = None,
                 in_import_allocation: Dict[str, Union[Tuple[str, str], Tuple[str, str, AllocationMethod],
                                                       List[Union[Tuple[str, str], Tuple[str, str, AllocationMethod]]]]] = None,
                 do_only_allocation: bool = False,
                 do_compare_adaptation: bool = False,
                 do_adapt_to_everything: bool = False,
                 do_rerank_learners: bool = True):
        
        self.key_target = in_key_target
        self.keys_features = in_keys_features
        self.do_exclude = do_exclude

        self.tags_allocation = in_tags_allocation

        self.do_immediate_responses = do_immediate_responses
        self.do_adapt_to_everything = do_adapt_to_everything
        self.do_rerank_learners = do_rerank_learners

        # How to handle imported pipelines, if any.
        self.directory_import = in_directory_import
        self.import_allocation = in_import_allocation
        self.do_only_allocation = do_only_allocation
        self.do_compare_adaptation = do_compare_adaptation

# TODO: Clean-up labels around tags and keys.
class ProblemSolution:
    """
    A container for all learners currently in production.
    """
    def __init__(self, in_instructions: ProblemSolverInstructions, in_strategy: Strategy, 
                 in_data_storage: DataStorage,
                 in_directory_results: str = None):

        self.directory = in_directory_results
        self.prepare_results()

        # Track pipelines that have been previously been successfully inserted into this solution.
        self.inserted_pipelines = dict()

        # Define a group as a champion and multiple challengers learning on a particular subset of data.
        # Define a filter as the rules for generating that subset of data.
        # Both dicts have the same keys.
        self.groups = dict()
        self.filters = dict()

        # If there are no allocation instructions, there are no filters.
        # One group learns on all data.
        self.id_no_filter = ""
        self.groups[self.id_no_filter] = list()
        self.filters[self.id_no_filter] = None

        # Examine how the user specifies data should be allocated between groups.
        tags_allocation = in_instructions.tags_allocation
        if not tags_allocation is None:
            for key_allocation in tags_allocation:
                if not isinstance(key_allocation, tuple):
                    key_allocation = (key_allocation, AllocationMethod.ONE_EACH)
                key_data = key_allocation[0]
                method_allocation = key_allocation[-1]

                unique_values = in_data_storage.get_tag_values(key_data)
                if method_allocation == AllocationMethod.LEAVE_ONE_OUT:
                    if len(unique_values) == 1:
                        text_warning = ("Skipping leave-one-out filter definitions based on '%s'. "
                                        "Only one category found: %s" % (key_data, unique_values[0]))
                        log.warning("%s - %s" % (Timestamp(), text_warning))
                        continue

                # Each separate allocation key denotes another dimension of dataset partitioning.
                # All previous groups are subdivided further.
                # TODO: Consider options for independent partitionings, i.e. no sub-splitting.
                for key_group in list(self.groups.keys()):
                    for value in unique_values:
                        if method_allocation == AllocationMethod.ONE_EACH:
                            if key_group == self.id_no_filter:
                                key_group_new = "(" + key_data + "==" + value + ")"
                            else:
                                key_group_new = key_group + "&(" + key_data + "==" + value + ")"
                        elif method_allocation == AllocationMethod.LEAVE_ONE_OUT:
                            if key_group == self.id_no_filter:
                                key_group_new = "(" + key_data + "!=" + value + ")"
                            else:
                                key_group_new = key_group + "&(" + key_data + "!=" + value + ")"
                        else:
                            raise NotImplementedError
                        
                        self.groups[key_group_new] = list()
                        if self.filters[key_group] is None:
                            self.filters[key_group_new] = [(key_data, value, method_allocation)]
                        else:
                            self.filters[key_group_new] = (self.filters[key_group] 
                                                           + [(key_data, value, method_allocation)])
                        
                    del self.groups[key_group]
                    del self.filters[key_group]

        self.n_challengers = in_strategy.n_challengers
        self.do_rerank_learners = in_instructions.do_rerank_learners

        log.info("%s - Prepared a ProblemSolution. Number of learner-groups: %i\n"
                 "%s   Each group champion can have up to %i challengers."
                 % (Timestamp(), len(self.groups),
                    Timestamp(None), self.n_challengers))
        

    # TODO: Consider improvements to the logic.
    def import_learners(self, in_instructions: ProblemSolverInstructions):
        """
        Load and insert all pipelines within a directory stored within instructions.
        Determine what their data filters are for adaptation.
        """

        dir_import = in_instructions.directory_import
        import_allocation = in_instructions.import_allocation
        do_only_allocation = in_instructions.do_only_allocation

        key_target = None
        keys_features = None

        if not dir_import is None:
            for filename in os.listdir(dir_import):
                filepath = os.path.join(dir_import, filename)
                pipeline = joblib.load(filepath)
                pipeline.name = "Imported_" + pipeline.name
                pipeline.clean_history()
                if in_instructions.do_compare_adaptation:
                    pipeline.is_static = True
                    pipeline.name += "_Static"
                    pipeline_alt = joblib.load(filepath)
                    pipeline_alt.name = "Imported_" + pipeline_alt.name + "_Adaptive"
                    pipeline_alt.is_static = False
                    pipeline_alt.clean_history()

                key_group = self.id_no_filter

                # Determine what data this imported pipeline should adapt on.
                is_pipeline_in_allocation = False
                if not import_allocation is None:
                    for substring, allocation in import_allocation.items():

                        if substring in filename:
                            is_pipeline_in_allocation = True

                            key_group = self.id_no_filter
                            filter_group = list()

                            if isinstance(allocation, tuple):
                                allocation = [allocation]

                            for tag_tuple in allocation:
                                key_data = tag_tuple[0]
                                value = tag_tuple[1]
                                if len(tag_tuple) == 3:
                                    method_allocation = tag_tuple[2]
                                else:
                                    method_allocation = AllocationMethod.ONE_EACH

                                if method_allocation == AllocationMethod.ONE_EACH:
                                    if key_group == self.id_no_filter:
                                        key_group = "(" + key_data + "==" + value + ")"
                                    else:
                                        key_group = key_group + "&(" + key_data + "==" + value + ")"
                                elif method_allocation == AllocationMethod.LEAVE_ONE_OUT:
                                    if key_group == self.id_no_filter:
                                        key_group = "(" + key_data + "!=" + value + ")"
                                    else:
                                        key_group = key_group + "&(" + key_data + "!=" + value + ")"
                                else:
                                    raise NotImplementedError
                                
                                filter_group.append((key_data, value, method_allocation))

                            # If the user has specified new filters, instantiate a group keyed to that filter.
                            if not key_group in self.groups:
                                self.groups[key_group] = list()
                                self.filters[key_group] = filter_group

                # TODO: Stop loading pipelines unnecessarily before rejecting them if doing only allocation.
                if (not do_only_allocation) or (do_only_allocation and (import_allocation is not None) and is_pipeline_in_allocation):
                    self.insert_learner(in_pipeline = pipeline, in_key_group = key_group)
                    if in_instructions.do_compare_adaptation:
                        self.insert_learner(in_pipeline = pipeline_alt, in_key_group = key_group)

                    # The last component of a pipeline is a predictor with a target.
                    # The first component of a pipeline will store initial features.
                    # Get both so that target/features can be imported.
                    key_target = pipeline.components[-1].key_target
                    keys_features = pipeline.components[0].keys_features

        # Return the last import pipeline's target/features.
        return key_target, keys_features


    # TODO: Consider whether to make a lack of re-ranking more meaningful, e.g. replacing poor performers in place.
    def insert_learner(self, in_pipeline: MLPipeline, in_key_group: str, do_replace: bool = False):
        """
        Insert a new learner into a group of learners.
        If there are too many challengers, remove the worst performer according to testing loss.
        Optionally, remove an existing pipeline with the same name, e.g. with an adapted pipeline.

        If the user decided not to rerank when declaring an intent to learn, there is no loss-based sorting.
        Warning: This means that ensembled solutions are arbitrary and new challengers cannot compete.
        """
        list_pipelines = self.groups[in_key_group]

        if do_replace:
            try:
                index_to_remove = next(i for i, pipeline in enumerate(list_pipelines) if pipeline.name == in_pipeline.name)
            except StopIteration:
                pass
            else:
                list_pipelines.pop(index_to_remove)

        list_pipelines.append(in_pipeline)
        if self.do_rerank_learners:
            self.groups[in_key_group] = sorted(list_pipelines, key=lambda p: p.get_loss())
        else:
            self.groups[in_key_group] = list_pipelines
        text_key_group = "" if in_key_group == self.id_no_filter else ": %s" % in_key_group
        log.info("%s - Learner Group%s -> %s" % (Timestamp(), text_key_group, 
                                                 ["%s: %0.2e" % (pipeline.name, pipeline.get_loss()) 
                                                  for pipeline in self.groups[in_key_group]]))
        
        pipeline_removed = None
        if len(self.groups[in_key_group]) > self.n_challengers + 1:
            pipeline_removed = self.groups[in_key_group].pop()
            log.info("%s   Removing uncompetitive challenger pipeline '%s' with loss: %0.2f"
                     % (Timestamp(None), pipeline_removed.name, pipeline_removed.get_loss()))
            
        # Record the pipeline if it was not immediately removed.
        if pipeline_removed is None or not pipeline_removed.name == in_pipeline.name:
            if not in_pipeline.name in self.inserted_pipelines:
                self.append_info_file(in_pipeline)
            self.inserted_pipelines[in_pipeline.name] = True
            
        
            
    def get_learners(self, in_key_group: str = None):

        list_learners = list()
        if in_key_group is None:
            for group in self.groups.values():
                list_learners.extend(group)
        else:
            list_learners = self.groups[in_key_group]
            
        return list_learners
    


    def count_learners(self):
        amount = 0
        for group in self.groups.values():
            amount += len(group)
        return amount


    # TODO: Reconsider which objects should be responsible for preparing results.
    def prepare_results(self):
        """
        Delete any pre-existing results folder and create a new one.
        Start up a new file for describing pipelines.
        """
        if self.directory is None:
            prefix = "./results/"
        else:
            prefix = self.directory + "/results/"
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        os.makedirs(prefix)

        filepath = prefix + "info_pipelines.txt"
        with open(filepath, "w") as file:
            pass

    # TODO: Consider if there is a feasible way to make the info sorted.
    def append_info_file(self, in_pipeline: MLPipeline):
        if self.directory is None:
            prefix = "./results/"
        else:
            prefix = self.directory + "/results/"
        filepath = prefix + "info_pipelines.txt"

        with open(filepath, "a") as file:
            file.write("%s: %s; Initial Loss: %s\n" % (in_pipeline.name, in_pipeline.components_as_string(do_hpars = True),
                                                       in_pipeline.get_loss()))

    def export_learners(self):
        if self.directory is None:
            prefix = "./pipelines/"
        else:
            prefix = self.directory + "/pipelines/"
        os.makedirs(prefix, exist_ok = True)

        for key_group, pipelines in self.groups.items():
            string_group = key_group
            if not string_group == "":
                string_group = "_" + string_group
            for rank, pipeline in enumerate(pipelines):
                filepath = prefix + "L" + str(rank) + string_group + "_" + pipeline.name + ".pipe"
                print(pipeline)
                print(filepath)
                joblib.dump(pipeline, filepath)