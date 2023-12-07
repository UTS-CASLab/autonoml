# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:20:56 2023

@author: David J. Kedziora
"""

from enum import Enum

from sklearn import metrics

class LossFunction(Enum):
    RMSE = "RMSE"
    ZERO_ONE = "Zero-One Classification Loss"

    @classmethod
    def default(cls):
        return cls.RMSE
    
    @classmethod
    def from_string(cls, in_string):
        for member in cls:
            if member.value == in_string:
                return member
        raise ValueError("No matching LossFunction for string: %s" % in_string)

def calculate_loss(y_response, y_true, in_loss_function: LossFunction = None):
        """
        Compare a set of predictions against a set of expected values.
        The output must be a loss.
        """
        if in_loss_function == LossFunction.RMSE:
            val = metrics.mean_squared_error(y_true = y_true, y_pred = y_response, 
                                             squared = False)
        elif in_loss_function == LossFunction.ZERO_ONE:
            val = metrics.zero_one_loss(y_true = y_true, y_pred = y_response)
        else:
            raise NotImplementedError
        
        return val