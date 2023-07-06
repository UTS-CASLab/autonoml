# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml
import os

# filename_data = "./data/mixed_0101_abrupto.csv"
dir_data = "./data"
filename_prefix = "sps_quality_1000_events"

proj = aml.AutonoMachine()
for filename in os.listdir(dir_data):
    if filename.startswith(filename_prefix):
        filepath = os.path.join(dir_data, filename)
        print(filepath)
        proj.ingest_file(in_filepath = os.path.join(dir_data, filename))
# proj.load_strategy()
# proj.learn(in_key_target = "class")
proj.data_storage.info()
proj.data_storage.get_dataframe()
proj.stop()