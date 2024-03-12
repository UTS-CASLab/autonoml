# -*- coding: utf-8 -*-
"""
Custom components wrapping around implementations by...
The Statistical Learning and Motor Control Group, The University of Edinburgh.

Created on Tue Feb  6 18:51:27 2024

@author: Thanh T. Khuat, David J. Kedziora
"""

from ..hyperparameter import HPFloat, HPCategorical
from ..component import MLOnlineLearner, MLRegressor
from ..data import DataFormatX, DataFormatY
from ..utils import CustomBool

import lwpr
import numpy as np

import os
import base64
from typing import List


# class WrapperLWPR(lwpr.LWPR):
#     """
#     The lwpr package is built in C and is not automatically pickled.
#     This class wraps around the LWPR model object and hacks in a pickling method.
#     Basically, a binary file is exported and its filepath is pickled.
#     """
#     def __init__(self, *args, in_name_component = None, in_filepath = None, **kwargs):
#         self.name_component = in_name_component
#         if not in_filepath is None:
#             # Load from binary file, if available.
#             super().__init__(in_filepath)
#         else:
#             # Otherwise, initialise as usual.
#             super().__init__(*args, **kwargs)

#     def __reduce__(self):
#         filepath = "./" + self.name_component + ".bin"
#         self.write_binary(filepath)

#         # Return a tuple with the class constructor and the arguments needed to reconstruct the object.
#         return (self.__class__, (), {"in_name_component": self.name_component, "in_filepath": filepath})



# class WrapperLWPR():
#     """
#     The lwpr package is built in C and is not automatically pickled.
#     This class wraps around the LWPR model object and hacks in a pickling method.
#     Basically, a binary file is exported/imported as needed and converted between a bytestring.
#     This allows deepcopy to keep things in memory, while joblib dump/load writes up a pickle file.
#     """
#     def __init__(self, *args, **kwargs):
#         in_bytestring = kwargs.get("in_bytestring", None)
#         print("test")
#         print(in_bytestring)
#         if not in_bytestring is None:
#             print("Import")
#             print(in_bytestring)
#             with open("./temp_lwpr.bin", "wb") as file:
#                 lwpr_bytes = base64.b64decode(in_bytestring)
#                 file.write(lwpr_bytes)
#             self.model = lwpr.LWPR("./temp_lwpr.bin")
#             os.remove("./temp_lwpr.bin")
#         else:
#             # Otherwise, initialise as usual.
#             self.model = lwpr.LWPR(*args, **kwargs)

#     def __getattr__(self, id_attribute):
#         # Delegate any attribute access to the wrapped LWPR instance.
#         return getattr(self.model, id_attribute)

#     def __reduce__(self):
#         self.model.write_binary("./temp_lwpr.bin")
#         with open("./temp_lwpr.bin", "rb") as file:
#             lwpr_bytes = file.read()
#         lwpr_data = base64.b64encode(lwpr_bytes).decode("utf-8")
#         os.remove("./temp_lwpr.bin")

#         print((self.__class__, (), {"in_bytestring": lwpr_data}))

#         # Return a tuple with the class constructor and the arguments needed to reconstruct the object.
#         return (self.__class__, (), {"in_bytestring": lwpr_data})



class WrapperLWPR():
    """
    The lwpr package is built in C and is not automatically pickled.
    This class wraps around the LWPR model object and hacks in a pickling method.
    Basically, a binary file is exported/imported as needed and converted between a bytestring.
    This allows deepcopy to keep things in memory, while joblib dump/load writes up a pickle file.
    """
    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            self.model = None
        else:
            self.model = lwpr.LWPR(*args, **kwargs)

    def __getattr__(self, id_attribute):
        # Delegate any attribute access to the wrapped LWPR instance.
        return getattr(self.model, id_attribute)

    def __getstate__(self):
        self.model.write_binary("./temp_lwpr.bin")
        with open("./temp_lwpr.bin", "rb") as file:
            lwpr_bytes = file.read()
        lwpr_data = base64.b64encode(lwpr_bytes).decode("utf-8")
        os.remove("./temp_lwpr.bin")

        state = {"bytestring": lwpr_data}
        return state
    
    def __setstate__(self, state):
        bytestring = state["bytestring"]
        with open("./temp_lwpr.bin", "wb") as file:
            lwpr_bytes = base64.b64decode(bytestring)
            file.write(lwpr_bytes)
        self.model = lwpr.LWPR("./temp_lwpr.bin")
        os.remove("./temp_lwpr.bin")



class LocallyWeightedProjectionRegressor(MLRegressor, MLOnlineLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.name += "_CustomSMLC_LWPR"
        self.format_x = DataFormatX.NUMPY_ARRAY_2D
        self.format_y = DataFormatY.NUMPY_ARRAY_2D
        self.is_setup_complete = False

    @staticmethod
    def new_hpars():
        hpars = dict()
        info = ("Initial distance metric for newly created receptive fields.")
        hpars["init_D"] = HPFloat(in_default = 1.0, in_min = 0.1, in_max = 10.0,
                                  is_log_scale = True, in_info = info)
        
        info = ("Whether or not distance metric is updated.")
        options = ["true", "false"]
        hpars["update_D"] = HPCategorical(in_options = options, in_info = info)

        info = ("Whether distance metric updates use second-order learning-rate adaptation "
                "via Incremental Delta Bar Delta (IDBD) algorithm.")
        options = ["true", "false"]
        hpars["meta"] = HPCategorical(in_options = options, in_info = info)
        
        info = ("Regularisation penalty multiplication factor.")
        hpars["penalty"] = HPFloat(in_default = 1.0e-6, in_min = 1.0e-7, in_max = 1.0e-5,
                                   is_log_scale = True, in_info = info)

        info = ("Weight activation threshold for triggering a new model.")
        hpars["w_gen"] = HPFloat(in_default = 0.1, in_min = 0.1, in_max = 0.9,
                                 in_info = info)
        
        return hpars

    def learn(self, x, y):
        if not self.is_setup_complete:
            raise Exception(f"Delayed setup for {self.name} was not completed before methods were called.")
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        for x_i, y_i in zip(x, y):
            print(type(x_i))
            print(type(y_i))
            self.model.update(x_i, y_i)

    def query(self, x):
        if not self.is_setup_complete:
            raise Exception(f"Delayed setup for {self.name} was not completed before methods were called.")
        x = x.astype(np.float64)
        responses = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            x_i = x[i, :]
            y_i = self.model.predict(x_i)
            responses[i, 0] = y_i
        print(responses)
        return responses
    
    def set_keys_features(self, in_keys_features: List[str]):
        """
        Custom utility function overwrite.
        This is required for updating the model with the number of features to expect.
        """
        self.keys_features = in_keys_features
        self.run_delayed_setup()

    def run_delayed_setup(self):
        n_f = len(self.keys_features)
        self.model = WrapperLWPR(n_f, 1)
        self.model.init_D = self.hpars["init_D"].val * np.eye(n_f)
        self.model.init_alpha = 100 * np.ones([n_f, n_f])
        self.model.w_gen = self.hpars["w_gen"].val
        self.model.penalty = self.hpars["penalty"].val
        self.model.update_D = bool(CustomBool(self.hpars["update_D"].val))
        self.model.meta = bool(CustomBool(self.hpars["meta"].val))
        self.is_setup_complete = True