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

from .data_storage import DataCollection

from copy import deepcopy

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

        self.training_y_true = list()
        self.training_y_response = list()
        self.testing_y_true = list()
        self.testing_y_response = list()

    def components_as_string(self, do_hpars = False):
        if do_hpars:
            text = " -> ".join([component.name + "(" + component.hpars_as_string() + ")" 
                                for component in self.components])
        else:
            text = " -> ".join([component.name for component in self.components])
        return text

    def process(self, x, y, in_format_x = None, in_format_y = None, 
                do_learn = True, do_query = True, do_score = True,
                do_remember = False, for_training = False):

        # Assume data is provided from storage in default format.
        # It will be reformatted as required by components.
        format_x = in_format_x
        if format_x is None:
            format_x = DataFormatX(0)
        format_y = in_format_y
        if format_y is None:
            format_y = DataFormatY(0)

        response = None
        metric = None

        num_components = len(self.components)
        for idx_component in range(num_components):

            # Every component learns from the incoming features and possibly the target.
            component = self.components[idx_component]
            x, format_x = component.reformat_x(x = x, in_format_old = format_x)
            y, format_y = component.reformat_y(y = y, in_format_old = format_y)
            if do_learn:
                component.learn(x=x, y=y)

            if isinstance(component, MLPreprocessor):
                # Preprocessing components modify the propagated feature set.
                x = component.transform(x=x)
            elif isinstance(component, MLPredictor):
                # TODO: Pipe predictions to the next component as a feature.
                pass
                # Any predictors in the pipeline are queried and scored.
                if do_query:
                    response = component.query(x=x)
                if do_score:
                    metric = component.score(x=x, y=y)

        # If there is a response, reformat it to the standard data format.
        if do_query:
            response = reformat_y(in_data = response,
                                  in_format_old = format_y,
                                  in_format_new = DataFormatY(0))

            # Memorise the true target and the final response.
            if do_remember:
                y = reformat_y(in_data = y,
                               in_format_old = format_y,
                               in_format_new = DataFormatY(0))

                # TODO: Decide if do_learn and for_training are identical.
                if for_training:
                    self.training_y_true.extend(y)
                    self.training_y_response.extend(response)
                else:
                    self.testing_y_true.extend(y)
                    self.testing_y_response.extend(response)

        # The final predictor in the pipeline has its response/score returned.
        return response, metric

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
                     in_data_collection: DataCollection,
                     in_info_process,
                     in_frac_data: float = 1.0,
                     for_training: bool = False):
    """
    A wrapper for pipeline training to be called from a ProblemSolver or elsewhere.
    This is designed for multiprocessing.
    """
    
    keys_features = in_info_process["keys_features"]
    key_target = in_info_process["key_target"]

    time_start = Timestamp().time
    x, y = in_data_collection.get_data(in_keys_features = keys_features,
                                       in_key_target = key_target,
                                       in_fraction = in_frac_data)
    time_end = Timestamp().time
    duration_prep = time_end - time_start

    time_start = Timestamp().time
    _, metric = in_pipeline.process(x, y, do_learn = for_training, do_remember = True, for_training = for_training)
    time_end = Timestamp().time
    duration_proc = time_end - time_start

    in_info_process["metric"] = metric
    in_info_process["duration_prep"] = duration_prep
    in_info_process["duration_proc"] = duration_proc

    return in_pipeline, in_info_process

def train_pipeline(in_pipeline: MLPipeline, in_data_collection: DataCollection,
                   in_info_process, in_frac_data: float = 1.0):
    return process_pipeline(in_pipeline, in_data_collection, in_info_process, in_frac_data, for_training = True)

def test_pipeline(in_pipeline: MLPipeline, in_data_collection: DataCollection, 
                  in_info_process):
    return process_pipeline(in_pipeline, in_data_collection, in_info_process)