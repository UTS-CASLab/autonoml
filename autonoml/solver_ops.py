# -*- coding: utf-8 -*-
"""
Additional operations that a ProblemSolver may enact as part of its processing.
These functions need to be in a separate module in case of multiprocessing.

Created on Thu Nov  9 11:02:45 2023

@author: David J. Kedziora
"""

from .data_storage import DataCollection, DataCollectionXY, SharedMemoryManager
from .solution import AllocationMethod
from .pipeline import MLPipeline, train_pipeline, test_pipeline

from typing import List, Dict

from copy import deepcopy
from itertools import chain

import pyarrow as pa
from pyarrow import csv as pacsv

import os



def get_collection(in_dict_observations, in_tag_to_collection_ids, 
                   in_tags_inclusive = None, in_tags_exclusive = None,
                   do_compression: bool = False):
    """
    Returns a concatenation of data collections linked/unlinked to 'inclusive/exclusive' tags.
    If there are no inclusive tags, all collections are selected prior to exclusions.
    Note: This function is external to DataStorage to enable multiprocessing.
    """
    if in_tags_inclusive is None:
        in_tags_inclusive = dict()
    if in_tags_exclusive is None:
        in_tags_exclusive = dict()
    
    if len(in_tags_inclusive) == 0:
        set_collection_ids = set(in_dict_observations.keys())
    else:
        set_collection_ids = set()

    for key_tag, value_tag in in_tags_inclusive.items():
        set_collection_ids = set_collection_ids | in_tag_to_collection_ids[key_tag][value_tag]
    for key_tag, value_tag in in_tags_exclusive.items():
        set_collection_ids = set_collection_ids - in_tag_to_collection_ids[key_tag][value_tag]

    # Concatenate the collections.
    data = pa.concat_tables([in_dict_observations[collection_id].data 
                             for collection_id in list(set_collection_ids)], 
                            promote = True)
    ids = list(chain(*(in_dict_observations[collection_id].ids 
                       for collection_id in list(set_collection_ids))))
    timestamps = list(chain(*(in_dict_observations[collection_id].timestamps
                              for collection_id in list(set_collection_ids))))
    collection = DataCollection(in_data = data, in_ids = ids, in_timestamps = timestamps)

    if do_compression:
        collection.data.combine_chunks()

    return collection



def filter_observations(in_dict_observations: Dict[int, DataCollection], 
                        in_tag_to_collection_ids, in_filter = None):
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

def prepare_data(in_collection: DataCollection, in_info_process, 
                 in_frac_validation: float = 0.25, in_n_sets: int = 1):

    # Prepare x and y at this stage to minimise data manipulation during training/testing.
    keys_features = in_info_process["keys_features"]
    key_target = in_info_process["key_target"]
    collection = in_collection.prepare_xy(in_keys_features = keys_features, in_key_target = key_target)

    sets_training = list()
    sets_validation = list()

    # TODO: Let users decide how many training/validation pairs to form.
    for _ in range(in_n_sets):
        set_validation, set_training = collection.split_randomly_by_fraction(in_fraction = in_frac_validation)
        sets_training.append(set_training)
        sets_validation.append(set_validation)

    return collection, sets_training, sets_validation



def develop_pipeline(in_pipeline: MLPipeline,
                     in_data_sharer: SharedMemoryManager,
                     in_info_process):

    in_observations, in_sets_training, in_sets_validation = in_data_sharer.load_observations()
    
    losses = list()

    info_process_clone = deepcopy(in_info_process)
    for set_training, set_validation in zip(in_sets_training, in_sets_validation):

        pipeline_clone = deepcopy(in_pipeline)

        # print("Initial Training Size: %i" % set_training.get_amount())
        pipeline_clone, _, _ = train_pipeline(in_pipeline = pipeline_clone,
                                              in_data_collection = set_training,
                                              in_info_process = info_process_clone)
        
        # print("Validation Size: %i" % set_validation.get_amount())
        pipeline_clone, _, _ = test_pipeline(in_pipeline = pipeline_clone,
                                             in_data_collection = set_validation,
                                             in_info_process = info_process_clone)
        
        losses.append(pipeline_clone.get_loss())

    loss = sum(losses)/len(losses)

    # print("Final Training Size: %i" % in_observations.get_amount())
    pipeline, _, _ = train_pipeline(in_pipeline = in_pipeline,
                                    in_data_collection = in_observations,
                                    in_info_process = in_info_process)
    
    # Short of further testing, its starting loss is the validation score it received here.
    pipeline.set_loss(loss)
    
    return pipeline, in_info_process



# TODO: Make destination for response outputs more modular, script-based, and user-controlled.

def anticipate_responses():
    prefix = "./results/"
    os.makedirs(prefix, exist_ok = True)

def get_responses(in_pipeline: MLPipeline,
                  in_queries: DataCollectionXY,
                  in_info_process):
    
    pipeline, responses, info_process = test_pipeline(in_pipeline = in_pipeline,
                                                      in_data_collection = in_queries,
                                                      in_info_process = in_info_process)
    
    return responses, pipeline, info_process

def action_responses(in_queries: DataCollectionXY):
    prefix = "./results/"
    filepath = prefix + "responses.csv"

    write_options = pacsv.WriteOptions(include_header = True)
    pacsv.write_csv(in_queries.quick_merge_xy().data, filepath, write_options = write_options)