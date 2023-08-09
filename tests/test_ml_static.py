# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml
import os

# filename_data = "./data/mixed_0101_abrupto.csv"
dir_data = "./data"
filename_substring = "sps_quality_1000_events"

proj = aml.AutonoMachine()
proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_1p2uW_3000cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_2p5uW_4000cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_4uW_4100cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_8uW_5100cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_10uW_6000cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_10uW_12000cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_20uW_7000cps.csv")
# proj.ingest_file(in_filepath = "./data/train_sps_quality_1000_events_30uW_7000cps.csv")
# for filename in os.listdir(dir_data):
#     if filename.startswith("train_" + filename_substring):
#         filepath = os.path.join(dir_data, filename)
#         proj.ingest_file(in_filepath = os.path.join(dir_data, filename))
# proj.load_strategy()
proj.info_storage()
proj.learn(in_key_target = "estimate",
           in_keys_features = ["best"], do_exclude = True)
aml.inspect_loop()
proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_1p2uW_3000cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_2p5uW_4000cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_4uW_4100cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_8uW_5100cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_10uW_6000cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_10uW_12000cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_20uW_7000cps.csv")
# proj.query_with_file(in_filepath = "./data/test_sps_quality_1000_events_30uW_7000cps.csv")
# for filename in os.listdir(dir_data):
#     if filename.startswith("test_" + filename_substring):
#         filepath = os.path.join(dir_data, filename)
#         proj.query_with_file(in_filepath = os.path.join(dir_data, filename))
# proj.data_storage.info()
# print(proj.data_storage.get_dataframe())
# proj.info_solver()
# proj.stop()

# pipe = proj.solver.pipelines["Pipe_2"]
# comp = pipe.components[-1]
# model = comp.model

# x, y = proj.data_storage.get_data(in_keys_features = comp.keys_features, 
#                                   in_key_target = comp.key_target,
#                                   in_format_x = comp.format_x, 
#                                   in_format_y = comp.format_y,
#                                   in_idx_start = -1)
# print(model.debug_one(x[-1]))