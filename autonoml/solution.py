# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:23 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pipeline import MLPipeline
from .strategy import Strategy

from .data_storage import DataStorage

from typing import List, Tuple, Union

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
                 in_tags_allocation: List[Union[str, Tuple[str, AllocationMethod]]] = None):
        
        self.key_target = in_key_target
        self.keys_features = in_keys_features
        self.do_exclude = do_exclude

        self.tags_allocation = in_tags_allocation

        self.do_immediate_responses = do_immediate_responses

# TODO: Clean-up labels around tags and keys.
class ProblemSolution:
    """
    A container for all learners currently in production.
    """
    def __init__(self, in_instructions: ProblemSolverInstructions, in_strategy: Strategy, 
                 in_data_storage: DataStorage):

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

        log.info("%s - Prepared a ProblemSolution. Number of learner-groups: %i\n"
                 "%s   Each group champion can have up to %i challengers."
                 % (Timestamp(), len(self.groups),
                    Timestamp(None), self.n_challengers))

    def insert_learner(self, in_pipeline: MLPipeline, in_key_group: str, do_replace: bool = False):
        """
        Insert a new learner into a group of learners.
        If there are too many challengers, remove the worst performer according to testing loss.
        Optionally, remove an existing pipeline with the same name, e.g. with an adapted pipeline.
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
        self.groups[in_key_group] = sorted(list_pipelines, key=lambda p: p.get_loss())
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


    # TODO: Reconsider which objects should be responsible for preparing results.
    def prepare_results(self):
        """
        Delete any pre-existing results folder and create a new one.
        Start up a new file for describing pipelines.
        """
        prefix = "./results/"
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        os.makedirs(prefix)

        filepath = prefix + "info_pipelines.txt"
        with open(filepath, "w") as file:
            pass

    # TODO: Consider if there is a feasible way to make the info sorted.
    def append_info_file(self, in_pipeline: MLPipeline):
        prefix = "./results/"
        filepath = prefix + "info_pipelines.txt"

        with open(filepath, "a") as file:
            file.write("%s: %s\n" % (in_pipeline.name, in_pipeline.components_as_string(do_hpars = True)))

    def export_learners(self):
        prefix = "./pipelines/"
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