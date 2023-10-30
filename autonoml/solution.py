# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:58:23 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .pipeline import MLPipeline, train_pipeline, test_pipeline
from .strategy import Strategy

from .data_storage import DataStorage, DataCollection, DataCollectionXY, SharedMemoryManager, get_collection

from typing import List, Dict
from enum import Enum
from copy import deepcopy

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
            


def filter_observations(in_dict_observations: Dict[int, DataCollection], 
                        in_tag_to_collection_ids, in_filter):
    """
    Based on allocation-specific filtering requirements, collate training data.
    """
    tags_inclusive = dict()
    tags_exclusive = dict()
    if not in_filter is None:
        for filter_spec in in_filter:
            key_filter = filter_spec[0]
            value_filter = filter_spec[1]
            allocation_method = filter_spec[2]
            if allocation_method == AllocationMethod.ONE_EACH:
                tags_inclusive[key_filter] = value_filter
            elif allocation_method == AllocationMethod.LEAVE_ONE_OUT:
                tags_exclusive[key_filter] = value_filter
    observations = get_collection(in_dict_observations = in_dict_observations, 
                                  in_tag_to_collection_ids = in_tag_to_collection_ids,
                                  in_tags_inclusive = tags_inclusive, in_tags_exclusive = tags_exclusive,
                                  do_compression = True)
        
        # # TODO: Decide what history of data to train pipelines on.
        # idx_start = 0
        # idx_stop = None
        # if idx_stop is None:
        #     idx_stop = observations.get_amount()

        # observations, _ = observations.split_by_range(in_idx_start = idx_start,
        #                                               in_idx_stop = idx_stop)

    return observations

def prepare_observations(in_observations: DataCollection, in_info_process, 
                         in_frac_validation: float, in_n_sets: int = 1):

    # Prepare x and y at this stage to minimise data manipulation during training.
    keys_features = in_info_process["keys_features"]
    key_target = in_info_process["key_target"]
    observations = in_observations.prepare_xy(in_keys_features = keys_features, in_key_target = key_target)

    sets_training = list()
    sets_validation = list()

    # TODO: Let users decide how many training/validation pairs to form.
    for idx_set in range(in_n_sets):
        set_validation, set_training = observations.split_randomly_by_fraction(in_fraction = in_frac_validation)
        sets_training.append(set_training)
        sets_validation.append(set_validation)

    return observations, sets_training, sets_validation



def develop_pipeline(in_pipeline: MLPipeline,
                     in_data_sharer: SharedMemoryManager,
                     in_observations: DataCollectionXY,
                     in_sets_training: List[DataCollectionXY], in_sets_validation: List[DataCollectionXY],
                     in_info_process):

    if not in_data_sharer is None:
        in_observations, in_sets_training, in_sets_validation = in_data_sharer.load_observations()
    
    losses = list()

    info_process_clone = deepcopy(in_info_process)
    for set_training, set_validation in zip(in_sets_training, in_sets_validation):

        pipeline_clone = deepcopy(in_pipeline)

        # print("Initial Training Size: %i" % set_training.get_amount())
        pipeline_clone, _ = train_pipeline(in_pipeline = pipeline_clone,
                                           in_data_collection = set_training,
                                           in_info_process = info_process_clone)
        
        # print("Validation Size: %i" % set_validation.get_amount())
        pipeline_clone, _ = test_pipeline(in_pipeline = pipeline_clone,
                                          in_data_collection = set_validation,
                                          in_info_process = info_process_clone)
        
        losses.append(pipeline_clone.get_loss())

    loss = sum(losses)/len(losses)

    # print("Final Training Size: %i" % in_observations.get_amount())
    pipeline, _ = train_pipeline(in_pipeline = in_pipeline,
                                 in_data_collection = in_observations,
                                 in_info_process = in_info_process)
    
    # Short of further testing, its starting loss is the validation score it received here.
    pipeline.set_loss(loss)
    
    return pipeline, in_info_process