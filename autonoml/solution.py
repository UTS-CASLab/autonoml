# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:23 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pipeline import MLPipeline
from .strategy import Strategy

from .data_storage import DataStorage

from enum import Enum

# TODO: Make these options clear to a user.
class AllocationMethod(Enum):
    """
    Descriptors for how to subdivide data per learner.
    """
    ONE_EACH = 0
    LEAVE_ONE_OUT = 1

class ProblemSolverInstructions:
    def __init__(self, in_key_target: str, in_keys_features = None, do_exclude: bool = False, 
                 in_strategy: Strategy = None, in_keys_allocation = None):
        
        self.key_target = in_key_target
        self.keys_features = in_keys_features
        self.do_exclude = do_exclude

        self.do_query_after_complete = True

        # Create a default strategy if user did not provide one.
        if in_strategy is None:
            self.strategy = Strategy()
        else:
            self.strategy = in_strategy

        # Defines how the solution should look.
        self.keys_allocation = in_keys_allocation
        self.n_challengers = 2

# TODO: Clean-up labels around tags and keys.
class ProblemSolution:
    """
    A container for all learners currently in production.
    """
    def __init__(self, in_instructions: ProblemSolverInstructions, in_data_storage: DataStorage):

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
        keys_allocation = in_instructions.keys_allocation
        if not keys_allocation is None:
            for key_allocation in keys_allocation:
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

        self.n_challengers = in_instructions.n_challengers

        log.info("%s - Prepared a ProblemSolution. Number of learner-groups: %i\n"
                 "%s   Each group champion can have up to %i challengers."
                 % (Timestamp(), len(self.groups),
                    Timestamp(None), self.n_challengers))

    def insert_learner(self, in_pipeline: MLPipeline, in_key_group: str):
        list_pipelines = self.groups[in_key_group]
        list_pipelines.append(in_pipeline)
        self.groups[in_key_group] = sorted(list_pipelines, key=lambda p: p.get_loss())
        log.info("%s - %s -> %s" % (Timestamp(), in_key_group, ["%s: %0.2f" % (pipeline.name, pipeline.get_loss())
                                                                for pipeline in self.groups[in_key_group]]))
        if len(self.groups[in_key_group]) > self.n_challengers + 1:
            pipeline_removed = self.groups[in_key_group].pop()
            log.info("%s   Removing uncompetitive challenger pipeline '%s' with loss: %0.2f"
                     % (Timestamp(None), pipeline_removed.name, pipeline_removed.get_loss()))