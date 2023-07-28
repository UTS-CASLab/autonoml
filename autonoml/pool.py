# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 17:16:42 2023

@author: David J. Kedziora
"""

from .utils import log, Timestamp
from .plot import plot_feature_importance, plot_performance
from .data import (DataFormatX, DataFormatY,
                   reformat_x, reformat_y)

from sklearn import (preprocessing, cross_decomposition,
                     dummy, svm)
                     
from river import linear_model
from river import metrics

# from sklearn.cross_decomposition import PLSRegression
# from sklearn.svm import LinearSVR

# import functools

# def deco_learn(learn):
#     @functools.wraps(learn)
#     def wrapper_decorator(self, x, y):
#         if self.model:
#             try:
#                 learn(self, x, y)
#             except Exception as e:
#                 raise(e)
#         else:
#             text_error = "Cannot fit an empty MLComponent to data."
#             log.error("%s - %s" % (Timestamp(), text_error))
#             raise Exception(text_error)
#     return wrapper_decorator

# def deco_score(score):
#     @functools.wraps(score)
#     def wrapper_decorator(self, x, y, do_remember = False, for_training = False):
#         if self.model:
#             if do_remember:
#                 if for_training:
#                     self.training_y_true.extend(y)
#                     self.training_y_response.extend(self.query(x))
#                 else:
#                     self.testing_y_true.extend(y)
#                     self.testing_y_response.extend(self.query(x))
#             try:
#                 value = score(self, x, y)
#             except Exception as e:
#                 raise(e)
#             return value
#         else:
#             text_error = "Cannot score an empty MLComponent on test data."
#             log.error("%s - %s" % (Timestamp(), text_error))
#             raise Exception(text_error)
#     return wrapper_decorator

# def deco_query(query):
#     @functools.wraps(query)
#     def wrapper_decorator(self, x):
#         if self.model:
#             try:
#                 response = query(self, x)
#             except Exception as e:
#                 raise(e)
#             return response
#         else:
#             text_error = "Cannot query an empty MLComponent on its response to data."
#             log.error("%s - %s" % (Timestamp(), text_error))
#             raise Exception(text_error)
#     return wrapper_decorator

# class MLComponent:
#     """
#     A base class for components of a learning pipeline.
#     """
    
#     count = 0
    
#     def __init__(self):
#         self.model = None
#         self.name = str(MLComponent.count)
#         MLComponent.count += 1
        
#         # # TODO: Decide the format that everything will be stored in. Native or converted.
#         # self.training_y_true = list()
#         # self.training_y_response = list()
#         # self.testing_y_true = list()
#         # self.testing_y_response = list()

#     def get_feature_importance(self):
#         return None

#     def inspect_structure(self, in_keys_features):
#         if self.model:
#             importance = self.get_feature_importance()
#             if not importance is None:
#                 plot_feature_importance(in_keys_features = in_keys_features,
#                                         in_importance = importance,
#                                         in_title = "Feature Importance: " + self.name)
#         else:
#             text_error = "Cannot inspect the structure of an empty MLComponent."
#             log.error("%s - %s" % (Timestamp(), text_error))
#             raise Exception(text_error)
            
#     def inspect_performance(self, for_training = False):
#         if self.model:
#             if for_training:
#                 text_type = "Training"
#                 vals_response = self.training_y_response
#                 vals_true = self.training_y_true
#             else:
#                 text_type = "Testing"
#                 vals_response = self.testing_y_response
#                 vals_true = self.testing_y_true

#             if self.data_format == DataFormat.LISTS_OF_X_LISTS_AND_Y_LISTS:
#                 # Flatten into lists that can be plotted.
#                 vals_response = [list(row) for row in zip(*vals_response)][0]
#                 vals_true = [list(row) for row in zip(*vals_true)][0]

#             plot_performance(in_vals_response = vals_response,
#                              in_vals_true = vals_true,
#                              in_title = "Performance (" + text_type + "): " + self.name)
                
#         else:
#             text_error = "Cannot inspect the performance of an empty MLComponent."
#             log.error("%s - %s" % (Timestamp(), text_error))
#             raise Exception(text_error)

class MLComponent:
    """
    A base class for components of a learning pipeline.
    """
    
    count = 0
    
    def __init__(self):
        self.model = None
        self.name = str(MLComponent.count)
        MLComponent.count += 1

        # Every component operates on input data organised in a certain format.
        # This data represents a set of features.
        self.keys_features = None
        self.data_format_x = None

    def learn(self, x, y):
        raise NotImplementedError
    
    def reformat_x(self, x, in_format_old):
        x = reformat_x(in_data = x, 
                       in_format_old = in_format_old, 
                       in_format_new = self.data_format_x,
                       in_keys_features = self.keys_features)
        return x, self.data_format_x
    
    def reformat_y(self, y, in_format_old):
        return y, in_format_old

class MLPreprocessor(MLComponent):
    def __init__(self):
        super().__init__()

    def transform(self, x):
        raise NotImplementedError

class MLPredictor(MLComponent):
    def __init__(self):
        super().__init__()

        # Predictors work with output data organised in a certain format.
        # This data represents a target variable.
        self.key_target = None
        self.data_format_y = None

    def score(self, x, y):
        raise NotImplementedError
    
    def query(self, x):
        raise NotImplementedError
    
    def reformat_y(self, y, in_format_old):
        y = reformat_y(in_data = y, 
                       in_format_old = in_format_old, 
                       in_format_new = self.data_format_y)
        return y, self.data_format_y

    def get_feature_importance(self):
        return None



#%% Scikit-learn components.

class SKLearnPreprocessor(MLPreprocessor):
    def __init__(self):
        super().__init__()
        self.name += "_SKLearnPreproc"
        self.data_format_x = DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES

class StandardScaler(SKLearnPreprocessor):
    def __init__(self):
        super().__init__()
        self.model = preprocessing.StandardScaler()
        self.name += "_StandardScalar"

    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    def transform(self, x):
        return self.model.transform(X=x)



class SKLearnPredictor(MLPredictor):
    def __init__(self):
        super().__init__()
        self.name += "_SKLearnPred"
        self.data_format_x = DataFormatX.LIST_OF_LISTS_ACROSS_FEATURES
        self.data_format_y = DataFormatY.LIST

    # @deco_learn
    def learn(self, x, y):
        self.model.fit(X=x, y=y)

    # @deco_score
    def score(self, x, y):
        return self.model.score(X=x, y=y)

    # @deco_query
    def query(self, x):
        return self.model.predict(X=x)
    
class DummyRegressor(SKLearnPredictor):
    def __init__(self):
        super().__init__()
        self.model = dummy.DummyRegressor()
        self.name += "_Dummy"
    
class PartialLeastSquaresRegressor(SKLearnPredictor):
    def __init__(self):
        super().__init__()
        self.model = cross_decomposition.PLSRegression(n_components=1)
        self.name += "_PLSR"
        self.data_format_y = DataFormatY.LIST_OF_LISTS_ACROSS_TARGETS

    # @deco_learn
    def learn(self, x, y):
        self.model.fit(X=x, Y=y)

    def get_feature_importance(self):
        return self.model.coef_.T[0]
    
class LinearSupportVectorRegressor(SKLearnPredictor):
    def __init__(self):
        super().__init__()
        self.model = svm.LinearSVR()
        self.name += "_LinearSVR"

    def get_feature_importance(self):
        return self.model.coef_



#%% River components.
        
class OnlineLinearRegressor(MLComponent):
    def __init__(self):
        super().__init__()
        self.model = linear_model.LinearRegression()
        self.name += "_LinearRegressor"
        self.data_format_x = DataFormatX.PANDAS_DATAFRAME
        self.data_format_y = DataFormatY.PANDAS_SERIES

    # @deco_learn
    def learn(self, x, y):
        self.model.learn_many(X=x, y=y)

    # @deco_score
    def score(self, x, y):
        # TODO: Generalise the metrics.
        metric = metrics.R2()
        for y_true, y_pred in zip(y, self.query(x)):
            metric.update(y_true, y_pred)
        value = metric.get()
        return value

    # @deco_query
    def query(self, x):
        return self.model.predict_many(X=x)
    


#%% Pipeline code.

# TODO: Upgrade to DAGs.
class MLPipeline:
    """
    A class for a sequential learning pipeline.
    The pipeline contains a list of learning components.
    The last must be a predictor.
    """

    count = 0

    def __init__(self, in_keys_features, in_key_target, in_components = None):
        self.name = str(MLPipeline.count)
        MLPipeline.count += 1

        log.info("%s - Constructing MLPipeline '%s'." % (Timestamp(), self.name))

        self.components = list()

        if in_components is None:
            # TODO: Implement and use DummyClassifier depending on the problem.
            in_components = [DummyRegressor()]

        for component in in_components:

            # Pass on the relevant variable keys to individual components.
            # TODO: Consider feature selection/generation down pipeline.
            component.keys_features = in_keys_features
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

    def components_as_string(self):
        return " -> ".join([component.name for component in self.components])

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

    # # @deco_learn
    # def learn(self, x, y, in_format_x = None, in_format_y = None):

    #     # Assume data is provided from storage in default format.
    #     # It will be reformatted as required by components.
    #     format_x = in_format_x
    #     if format_x is None:
    #         format_x = DataFormatX(0)
    #     format_y = in_format_y
    #     if format_y is None:
    #         format_y = DataFormatY(0)

    #     num_components = len(self.components)
    #     for idx_component in range(num_components):

    #         # Every component learns from the incoming features and possibly the target.
    #         component = self.components[idx_component]
    #         x, format_x = component.reformat_x(x = x, in_format_old = format_x)
    #         y, format_y = component.reformat_y(y = y, in_format_old = format_y)
    #         component.learn(x=x, y=y)

    #         if isinstance(component, MLPreprocessor):
    #             # Preprocessing components modify the propagated feature set.
    #             x = component.transform(x=x)
    #         elif isinstance(component, MLPredictor):
    #             # TODO: Pipe predictions to the next component as a feature.
    #             pass

    # # @deco_score
    # def score(self, x, y, in_format_x = None, in_format_y = None,
    #           do_remember = False, for_training = False):

    #     # Assume data is provided from storage in default format.
    #     # It will be reformatted as required by components.
    #     format_x = in_format_x
    #     if format_x is None:
    #         format_x = DataFormatX(0)
    #     format_y = in_format_y
    #     if format_y is None:
    #         format_y = DataFormatY(0)

    #     value = None
    #     num_components = len(self.components)
    #     for idx_component in range(num_components):
    #         component = self.components[idx_component]
    #         x, format_x = component.reformat_x(x = x, in_format_old = format_x)
    #         y, format_y = component.reformat_y(y = y, in_format_old = format_y)

    #         if isinstance(component, MLPreprocessor):
    #             # Preprocessing components modify the propagated feature set.
    #             x = component.transform(x=x)
    #         elif isinstance(component, MLPredictor):
    #             # TODO: Pipe predictions to the next component as a feature.
    #             # Any predictors in the pipeline are scored.
    #             value = component.score(x=x, y=y)

    #     if do_remember:
    #         if for_training:
    #             self.training_y_true.extend(y)
    #             self.training_y_response.extend(self.query(x))
    #         else:
    #             self.testing_y_true.extend(y)
    #             self.testing_y_response.extend(self.query(x))

    #     # The final predictor in the pipeline has its score returned.
    #     return value

    # # @deco_query
    # def query(self, x):
        
    #     response = None
    #     num_components = len(self.components)
    #     for idx_component in range(num_components):
    #         component = self.components[idx_component]
    #         x, _ = component.reformat_data(x=x, y=None)
    #         if isinstance(component, MLPreprocessor):
    #             # Preprocessing components modify the propagated feature set.
    #             x = component.transform(x=x)
    #         elif isinstance(component, MLPredictor):
    #             # TODO: Pipe predictions to the next component as a feature.
    #             # Any predictors in the pipeline are queried.
    #             response = component.query(x=x)

    #     # The final predictor in the pipeline has its response returned.
    #     return response

    def inspect_structure(self):
        for component in self.components:
            if isinstance(component, MLPredictor):
                importance = component.get_feature_importance()
                if not importance is None:
                    plot_feature_importance(in_keys_features = component.keys_features,
                                            in_importance = importance,
                                            in_title = "Feature Importance: " + component.name)
            
    def inspect_performance(self, for_training = False):
        if for_training:
            text_type = "Training"
            vals_response = self.training_y_response
            vals_true = self.training_y_true
        else:
            text_type = "Testing"
            vals_response = self.testing_y_response
            vals_true = self.testing_y_true

        title = "Performance (%s): %s\n%s" % (text_type, self.name, self.components_as_string())
        plot_performance(in_vals_response = vals_response,
                         in_vals_true = vals_true,
                         in_title = title)