# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 23:02:29 2023

@author: David J. Kedziora
"""

from typing import List

class ProcessInformation:
    """
    A small data structure containing variables to be passed between solver-related functions.
    """
    def __init__(self, in_keys_features: List[str] = None, in_key_target: str = None,
                 in_id_last_old: int = None, in_id_last_new: int = None):
        self.keys_features = in_keys_features
        self.key_target = in_key_target
        self.id_last_old = in_id_last_old
        self.id_last_new = in_id_last_new

        self.n_available = None
        self.n_instances = None
        self.duration_prep = None
        self.duration_proc = None

        self.text = None

    def set_n_available(self, in_val):
        """
        The total number of data points available, e.g. as observations.
        """
        self.n_available = in_val

    def set_n_instances(self, in_val):
        """
        The actual number of data points involved in the process.
        """
        self.n_instances = in_val

    def set_duration_prep(self, in_val):
        self.duration_prep = in_val

    def set_duration_proc(self, in_val):
        self.duration_proc = in_val

    def set_text(self, in_val):
        """
        Informative text to be displayed elsewhere.
        """
        self.text = in_val