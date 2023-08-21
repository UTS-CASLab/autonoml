# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:30:58 2023

@author: David J. Kedziora
"""

def plan_strategy():

    strategy = None

    return strategy

# from . import components
# from .component import MLComponent, MLPredictor, MLPreprocessor

# import pkgutil

# pool_preprocessors = dict()
# pool_predictors = dict()

# # Import all modules within the components folder.
# for importer, module_name, is_pkg in pkgutil.iter_modules(components.__path__):
#     full_module_name = "%s.%s" % (components.__name__, module_name)
#     loaded_module = importer.find_module(full_module_name).load_module(full_module_name)
        
#     # Find all classes in the loaded modules that are MLComponents.
#     for name, obj in vars(loaded_module).items():
#         if isinstance(obj, type) and issubclass(obj, MLComponent):

#             # Consider ones that are as deep in their hierarchy without becoming unselectable.
#             # Note: The False in getattr is the value if the attribute cannot be found.
#             if not getattr(obj, "is_unselectable", False):
#                 subclasses = obj.__subclasses__()
#                 if (len(subclasses) == 0 
#                     or all(getattr(subclass, "is_unselectable", False) for subclass in subclasses)):
                    
#                     if issubclass(obj, MLPreprocessor):
#                         pool_preprocessors[obj.__name__] = obj
#                     elif issubclass(obj, MLPredictor):
#                         pool_predictors[obj.__name__] = obj

# print(pool_preprocessors)
# print(pool_predictors)
