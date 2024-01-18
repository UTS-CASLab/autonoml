# -*- coding: utf-8 -*-
"""
Additional operations that a ProblemSolver may enact as part of its processing.
These functions need to be in a separate module in case of multiprocessing.

Created on Thu Nov  9 11:02:45 2023

@author: David J. Kedziora
"""

from .data_storage import DataCollection, DataCollectionXY, SharedMemoryManager
from .solution import AllocationMethod, ProblemSolution
from .pipeline import MLPipeline, train_pipeline, test_pipeline
from .instructions import ProcessInformation

from typing import List, Dict

from copy import deepcopy
from itertools import chain

import pyarrow as pa
from pyarrow import csv as pacsv

import os
import numpy as np
from collections import Counter



def get_collection(in_dict_observations, in_tag_to_collection_ids, 
                   in_tags_inclusive = None, in_tags_exclusive = None,
                   in_info_process: ProcessInformation = None,
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

    # Create the basis of an empty data collection.
    empty_schema = pa.schema([])
    data = pa.Table.from_batches([], schema=empty_schema)
    ids = list()
    timestamps = list()

    # Permissively extend the contents per collection that is selected.
    id_last_old = None if in_info_process is None else in_info_process.id_last_old
    id_last_new = None if in_info_process is None else in_info_process.id_last_new

    for collection_id in list(set_collection_ids):
        collection, _ = in_dict_observations[collection_id].split_by_special_range(in_id_start_exclusive = id_last_old,
                                                                                   in_id_stop_inclusive = id_last_new)
        data = pa.concat_tables([data, collection.data],
                                promote_options = "permissive")
        ids.extend(collection.ids)
        timestamps.extend(collection.timestamps)
        
    collection = DataCollection(in_data = data, in_ids = ids, in_timestamps = timestamps)

    if do_compression:
        collection.combine_chunks()

    return collection



def filter_observations(in_dict_observations: Dict[int, DataCollection], 
                        in_tag_to_collection_ids, in_filter = None,
                        in_info_process: ProcessInformation = None):
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
                                  in_info_process = in_info_process,
                                  do_compression = True)

    return observations

def prepare_data(in_collection: DataCollection, in_info_process: ProcessInformation, 
                 in_frac_validation: float = 0.25, in_n_sets: int = 1,
                 do_shuffle_original: bool = True):

    # No validation is needed for adaptation.
    if in_info_process.do_adapt:
        in_n_sets = 0

    # Prepare x and y at this stage to minimise data manipulation during training/testing.
    # For adaptation, this is just the last line.
    keys_features = in_info_process.keys_features
    key_target = in_info_process.key_target
    collection = in_collection.prepare_xy(in_keys_features = keys_features, in_key_target = key_target,
                                          do_last_only = in_info_process.do_adapt)

    sets_training = list()
    sets_validation = list()
    
    for idx_set in range(in_n_sets):
        set_validation, set_training = collection.split_randomly_by_fraction(in_fraction = in_frac_validation, 
                                                                             in_seed = idx_set)
        sets_training.append(set_training)
        sets_validation.append(set_validation)

    # Shuffle the original collection, which is normally what pipelines are finally trained on.
    if do_shuffle_original:
        collection, _ = collection.split_randomly_by_fraction(in_fraction = 1.0, in_seed = 0)

    return collection, sets_training, sets_validation



def develop_pipeline(in_pipeline: MLPipeline,
                     in_data_sharer: SharedMemoryManager,
                     in_info_process: ProcessInformation):

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

    if in_info_process.do_adapt:
        loss = in_pipeline.get_loss()
    else:
        if len(losses) == 0:
            loss = np.inf
        else:
            loss = sum(losses)/len(losses)

    # print("Final Training Size: %i" % in_observations.get_amount())
    pipeline, _, info_process = train_pipeline(in_pipeline = in_pipeline,
                                               in_data_collection = in_observations,
                                               in_info_process = in_info_process)
    
    # Short of further testing, its starting loss is the validation score it received here.
    # In the case of adaptation, it is the last loss it had.
    pipeline.set_loss(loss)
    
    return pipeline, info_process



# TODO: Make destination for response outputs more modular, script-based, and user-controlled.
# TODO: Consider moving 'magic values' to settings.

#%% Functions for processing new observations, i.e. adaptation.

def adapt_to_data(in_pipeline: MLPipeline,
                  in_observations: DataCollectionXY,
                  in_info_process: ProcessInformation):
    
    pipeline, responses, info_process = adapt_pipeline(in_pipeline = in_pipeline,
                                                       in_data_collection = in_observations,
                                                       in_info_process = in_info_process)
    
    return responses, pipeline, info_process

def track_dynamics(in_observations: DataCollectionXY,
                     in_results_dict,
                     in_key_group: str,
                     in_solution: ProblemSolution,
                     in_info_process: ProcessInformation):
    """
    Track the dynamics of a group of learners as they adapt to data.
    """
    filepath_prefix = "./results/"
    group_string = in_key_group
    if not group_string == "":
        group_string = "_" + group_string
    filepath = filepath_prefix + "dynamics" + group_string + ".csv"

    key_target = in_info_process.key_target
            
    # Construct a table from all the exportable information.
    table_export = in_observations.x.rename_columns(['F_' + col for col in in_observations.x.schema.names])
    table_export = table_export.add_column(0, "IDs", pa.array(in_observations.ids))
    # table_export = table_export.add_column(0, "Timestamps", pa.array(in_observations.timestamps))   # TODO: Convert to string.
    table_export = table_export.append_column("T_" + key_target, in_observations.y)
    for idx in range(1 + in_solution.n_challengers):
        header_prefix = "L%i" % idx
        for key_content in ["responses", "loss", "name"]:
            if not idx in in_results_dict:
                list_append = [None]*in_observations.get_amount()
            else:
                list_append = in_results_dict[idx][key_content]
            if key_content == "responses":
                key_content_for_header = "_" + key_target
            else:
                key_content_for_header = ":" + key_content
            table_export = table_export.append_column(header_prefix + key_content_for_header, pa.array(list_append))

    # print(table_export)

    if not os.path.isfile(filepath):
        write_options = pacsv.WriteOptions(include_header = True)
        file_options = "wb"
    else:
        write_options = pacsv.WriteOptions(include_header = False)
        file_options = "ab"

    with open(filepath, file_options) as file:
        with pacsv.CSVWriter(file, table_export.schema, write_options = write_options) as writer:
            writer.write_table(table_export)





#%% Functions for processing new queries.

def get_responses(in_pipeline: MLPipeline,
                  in_queries: DataCollectionXY,
                  in_info_process: ProcessInformation):
    
    pipeline, responses, info_process = test_pipeline(in_pipeline = in_pipeline,
                                                      in_data_collection = in_queries,
                                                      in_info_process = in_info_process)
    
    return responses, pipeline, info_process

# TODO: Expand on this.
def ensemble_responses(in_dict_responses):

    # Get the responses from the best-ranked learner of each group, i.e. rank 0.
    all_responses = list()
    for tags in in_dict_responses:
        if 0 in in_dict_responses[tags]:
            all_responses.append(in_dict_responses[tags][0]["responses"])

    # Calculate the average across all best-ranked learner responses.
    try:
        return np.mean(all_responses, axis = 0)
    except:
        pass

    # Failing that, return the modes of responses across all best-ranked learners.
    # TODO: If this is inefficient, seek other ways to calculate cross-axis modes.
    transposed_responses = zip(*all_responses)
    ensembled_responses = [Counter(column).most_common(1)[0][0] for column in transposed_responses]

    return ensembled_responses

def action_responses(in_queries: DataCollectionXY,
                     in_responses_best,
                     in_results_dict,
                     in_collection_tag_string: str,
                     in_solution: ProblemSolution,
                     in_info_process: ProcessInformation):
    """
    Do something with the responses returned by the solution.
    Currently exports to a file defined by how the query collection was tagged by the user.
    """
    filepath_prefix = "./results/"
    tag_string = in_collection_tag_string
    if not tag_string == "":
        tag_string = "_" + tag_string
    filepath = filepath_prefix + "responses" + tag_string + ".csv"

    key_target = in_info_process.key_target
            
    # Construct a table from all the exportable information.
    table_export = in_queries.x.rename_columns(['F_' + col for col in in_queries.x.schema.names])
    table_export = table_export.add_column(0, "IDs", pa.array(in_queries.ids))
    # table_export = table_export.add_column(0, "Timestamps", pa.array(in_queries.timestamps))   # TODO: Convert to string.
    table_export = table_export.append_column("T_" + key_target, in_queries.y)
    table_export = table_export.append_column("S_" + key_target, pa.array(in_responses_best))
    for tags in in_solution.groups.keys():
        tags_for_prefix = tags
        if not tags_for_prefix == "":
            tags_for_prefix = ":" + tags_for_prefix
        for idx in range(1 + in_solution.n_challengers):
            header_prefix = "L%i%s" % (idx, tags_for_prefix)
            for key_content in ["responses", "loss", "name"]:
                if not idx in in_results_dict[tags]:
                    list_append = [None]*in_queries.get_amount()
                else:
                    list_append = in_results_dict[tags][idx][key_content]
                if key_content == "responses":
                    key_content_for_header = "_" + key_target
                else:
                    key_content_for_header = ":" + key_content
                table_export = table_export.append_column(header_prefix + key_content_for_header, pa.array(list_append))

    # print(table_export)

    if not os.path.isfile(filepath):
        write_options = pacsv.WriteOptions(include_header = True)
        file_options = "wb"
    else:
        write_options = pacsv.WriteOptions(include_header = False)
        file_options = "ab"

    with open(filepath, file_options) as file:
        with pacsv.CSVWriter(file, table_export.schema, write_options = write_options) as writer:
            writer.write_table(table_export)