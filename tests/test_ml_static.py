# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml

filename_data = "./data/mixed_0101_abrupto.csv"

proj = aml.AutonoMachine()
proj.ingest_file(in_filename = filename_data)
# proj.load_strategy()
proj.learn(in_key_target = "class")