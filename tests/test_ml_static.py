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
proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_1p2uW_3000cps.csv")
proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_2p5uW_4000cps.csv")
# for filename in os.listdir(dir_data):
#     if filename.startswith(filename_prefix):
#         filepath = os.path.join(dir_data, filename)
#         print(filepath)
#         proj.ingest_file(in_filepath = os.path.join(dir_data, filename))
# proj.load_strategy()
proj.info_storage()
proj.learn(in_key_target = "best", 
           in_keys_features = ["estimate"], do_exclude = True)
proj.test_with_file(in_filepath = "./data/test_sps_quality_1000_events_1p2uW_3000cps.csv")
proj.test_with_file(in_filepath = "./data/test_sps_quality_1000_events_2p5uW_4000cps.csv")
# proj.data_storage.info()
# print(proj.data_storage.get_dataframe())
# proj.stop()