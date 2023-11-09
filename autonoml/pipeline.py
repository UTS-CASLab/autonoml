# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:13:43 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .plot import gen_fig_feature_importance, gen_fig_performance
from .data import DataFormatX, DataFormatY, reformat_y
from .component import MLPredictor, MLPreprocessor
from .components.sklearn import DummyRegressor      # TODO: Revisit this import.
from .metrics import LossFunction, calculate_loss

from .data_storage import DataCollection, DataCollectionXY

from typing import Union

from copy import deepcopy
from enum import Enum

import numpy as np

class PipelineProcessState(Enum):
    QUERY = 0
    LEARN = 1

# TODO: Upgrade to DAGs.
class MLPipeline:
    """
    A class for a sequential learning pipeline.
    The pipeline contains a list of learning components.
    The last must be a predictor.
    """

    count = 0   # Used only for naming unnamed pipelines.

    def __init__(self, in_keys_features, in_key_target, in_components = None,
                 in_name: str = None, do_increment_count: bool = True):
        if in_name is None:
            self.name = "Pipe_" + str(MLPipeline.count)
            if do_increment_count:
                MLPipeline.count += 1
        else:
            self.name = in_name

        log.info("%s - Constructing MLPipeline '%s'." % (Timestamp(), self.name))

        self.components = list()

        if in_components is None:
            # TODO: Implement and use DummyClassifier depending on the problem.
            in_components = [DummyRegressor()]

        for component in in_components:

            # Pass on the relevant variable keys to individual components.
            # TODO: Consider feature selection/generation down pipeline.
            component.keys_features = deepcopy(in_keys_features)
            if isinstance(component, MLPredictor):
                component.key_target = in_key_target

            self.components.append(component)
            log.info("%s   Attached MLComponent '%s'." 
                        % (Timestamp(None), component.name))

        if not isinstance(self.components[-1], MLPredictor):
            text_error = "The last component in an MLPipeline must be an MLPredictor."
            log.error("%s - %s" % (Timestamp(), text_error))
            raise Exception(text_error)

        # Maintain a history of the target variable as well as loss values.
        self.training_y_true = list()
        self.training_y_response = list()
        self.training_loss = np.inf
        self.testing_y_true = list()
        self.testing_y_response = list()
        self.testing_loss = np.inf

    def __repr__(self):
        return self.name + ": [" + self.components_as_string() + "]"

    def components_as_string(self, do_hpars = False):
        if do_hpars:
            text = " -> ".join([component.name + "(" + component.hpars_as_string() + ")" 
                                for component in self.components])
        else:
            text = " -> ".join([component.name for component in self.components])
        return text
    
    def get_loss(self, is_training: bool = False):
        if is_training:
            return self.training_loss
        else:
            return self.testing_loss
        
    def set_loss(self, in_loss, is_training: bool = False):
        if is_training:
            self.training_loss = in_loss
        else:
            self.testing_loss = in_loss

    # TODO: Use other non-RMSE metrics.
    # TODO: Decide whether loss should be calculated on smaller subsets of history.
    def update_loss(self, y_response, y_true, is_training: bool = False):
        """
        Update the loss stored by the pipeline, accessed easily as an attribute.
        Return the loss according to only the most recent values.
        """

        # It is possible that querying/learning is done without supervision.
        # Unsupervised instances have None as the target variable.
        y_response, y_true = zip(*[(element_response, element_true) 
                                   for element_response, element_true in zip(y_response, y_true) 
                                   if not element_true is None])
        
        loss_recent = np.inf

        # Loss only updates if there are new entries.
        if len(y_true) > 0:
            if is_training:
                self.training_y_true.extend(y_true)
                self.training_y_response.extend(y_response)
                self.training_loss = calculate_loss(y_response = self.training_y_response, 
                                                    y_true = self.training_y_true,
                                                    in_loss_function = LossFunction.RMSE)
            else:
                self.testing_y_true.extend(y_true)
                self.testing_y_response.extend(y_response)
                self.testing_loss = calculate_loss(y_response = self.testing_y_response,
                                                   y_true = self.testing_y_true,
                                                   in_loss_function = LossFunction.RMSE)
                
            loss_recent = calculate_loss(y_response = y_response, y_true = y_true,
                                         in_loss_function = LossFunction.RMSE)
        
        return loss_recent

    def process(self, x, y, do_query = False, do_learn = False):
        """
        Given a mapping of x to y, this function can process a pipeline in several ways:
        - Query on x and score the response against y.
        - Learn on x and y.
        - Do both, a.k.a. streamed learning.
        
        Both x and y are assumed to be in default formats DataFormatX(0) and DataFormatY(0).
        At the end, responses are converted to DataFormatY.LIST.
        """

        format_x = DataFormatX(0)
        format_y = DataFormatY(0)

        responses = None

        num_components = len(self.components)

        # Querying comes before learning in typical streaming ML.
        for state in [PipelineProcessState.QUERY, PipelineProcessState.LEARN]:

            # Query or learn only as required.
            if (state == PipelineProcessState.QUERY and do_query) or (state == PipelineProcessState.LEARN and do_learn):

                # Memorise a copy of the original feature space if querying and learning.
                # Note: No need to memorise the target space unless modifications are allowed.
                if (state == PipelineProcessState.QUERY and do_learn):
                    x_mem = deepcopy(x)

                for idx_component in range(num_components):

                    # The data is reformatted for the requirements of each component.
                    component = self.components[idx_component]
                    # time_start = Timestamp().time
                    x, format_x = component.reformat_x(x = x, in_format_old = format_x)
                    y, format_y = component.reformat_y(y = y, in_format_old = format_y)
                    # print("a %s" % (Timestamp().time - time_start))

                    if state == PipelineProcessState.LEARN:
                        # time_start = Timestamp().time
                        component.learn(x=x, y=y)
                        # print("b %s" % (Timestamp().time - time_start))

                    if isinstance(component, MLPreprocessor):

                        # Preprocessing components modify the propagated feature set.
                        x = component.transform(x=x)

                    elif isinstance(component, MLPredictor):

                        # Predictors respond to queries.
                        # TODO: Expand the feature set with the response.
                        predictions = component.query(x=x)

                        if idx_component == num_components - 1:
                            # Reformat the final response to a list format.
                            predictions = reformat_y(in_data = predictions,
                                                    in_format_old = format_y,
                                                    in_format_new = DataFormatY.LIST)
                            y = reformat_y(in_data = y,
                                        in_format_old = format_y,
                                        in_format_new = DataFormatY.LIST)
                            
                            if state == PipelineProcessState.QUERY:
                                responses = predictions
                                self.update_loss(y_response = predictions, y_true = y)
                            else:
                                self.update_loss(y_response = predictions, y_true = y, 
                                                 is_training = True)

                # Recall the original feature space if querying and learning.
                if (state == PipelineProcessState.QUERY and do_learn):
                    x = x_mem

        return responses

    def inspect_structure(self):
        """
        Generate a list of figures that inspect pipeline structure, if available.
        """
        figs = list()
        for component in self.components:
            if isinstance(component, MLPredictor):
                importance = component.get_feature_importance()
                if not importance is None:
                    fig = gen_fig_feature_importance(in_keys_features = component.keys_features,
                                                     in_importance = importance,
                                                     in_title = "Feature Importance: " + component.name)
                    figs.append(fig)
        return figs
            
    def inspect_performance(self, for_training = False):
        """
        Generate a list of figures that inspect pipeline performance, if available.
        """
        figs = list()

        if for_training:
            text_type = "Training"
            vals_response = self.training_y_response
            vals_true = self.training_y_true
        else:
            text_type = "Testing"
            vals_response = self.testing_y_response
            vals_true = self.testing_y_true

        if len(vals_response) > 0 and len(vals_true) > 0:
            title = "Performance (%s): %s\n%s" % (text_type, self.name, 
                                                  self.components_as_string(do_hpars = True))
            fig = gen_fig_performance(in_vals_response = vals_response,
                                    in_vals_true = vals_true,
                                    in_title = title)
            figs.append(fig)

        return figs
    


def process_pipeline(in_pipeline: MLPipeline,
                     in_data_collection:  DataCollectionXY, #Union[DataCollection, DataCollectionXY],
                     in_info_process,
                     in_frac_data: float = 1.0,
                     do_query: bool = False,
                     do_learn: bool = False):
    """
    A wrapper for pipeline training to be called from a ProblemSolver or elsewhere.
    This is designed for multiprocessing.
    """

    time_start = Timestamp().time
    # if isinstance(in_data_collection, DataCollection):
    #     keys_features = in_info_process["keys_features"]
    #     key_target = in_info_process["key_target"]
    #     x, y = in_data_collection.get_data(in_keys_features = keys_features,
    #                                     in_key_target = key_target,
    #                                     in_fraction = in_frac_data)
    if isinstance(in_data_collection, DataCollectionXY):
        x, y = in_data_collection.get_data(in_fraction = in_frac_data)
    else:
        raise NotImplementedError
    time_end = Timestamp().time
    duration_prep = time_end - time_start
    # print(duration_prep)

    time_start = Timestamp().time
    responses = in_pipeline.process(x, y, do_query = do_query, do_learn = do_learn)
    time_end = Timestamp().time
    duration_proc = time_end - time_start
    # print(duration_proc)

    in_info_process["duration_prep"] = duration_prep
    in_info_process["duration_proc"] = duration_proc
    in_info_process["n_instances"] = len(y)

    return in_pipeline, responses, in_info_process

def train_pipeline(in_pipeline: MLPipeline, in_data_collection: DataCollectionXY, #Union[DataCollection, DataCollectionXY],
                   in_info_process, in_frac_data: float = 1.0):
    return process_pipeline(in_pipeline, in_data_collection, in_info_process, in_frac_data, 
                            do_learn = True)

def test_pipeline(in_pipeline: MLPipeline, in_data_collection:  DataCollectionXY, #Union[DataCollection, DataCollectionXY], 
                  in_info_process):
    return process_pipeline(in_pipeline, in_data_collection, in_info_process,
                            do_query = True)