# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:52:35 2023

@author: David J. Kedziora
"""

import autonoml as aml

dir_data = "./data/sps"
filename_substring = "sps_quality_1000_events"
experimental_contexts = ["1p2uW_3000cps",
                         "2p5uW_4000cps",
                         "4uW_4100cps",
                         "8uW_5100cps",
                         "10uW_6000cps",
                         "10uW_12000cps",
                         "20uW_7000cps",
                         "30uW_7000cps",
                         None]

if __name__ == '__main__':

    strategy = aml.import_strategy("./test_ml_static.strat")

    proj = aml.AutonoMachine(do_mp = False)

    for experimental_context in experimental_contexts:
        if not experimental_context is None:
            proj.ingest_file(in_filepath = "%s/train_%s_%s.csv" % (dir_data, filename_substring, 
                                                                   experimental_context), 
                             in_tags = {"context": experimental_context})
            
    proj.info_storage()

    proj.learn(in_key_target = "best",
               in_keys_features = ["estimate"], do_exclude = True,
               in_strategy = strategy,
               in_tags_allocation = [("context", aml.AllocationMethod.LEAVE_ONE_OUT)])
    
    for experimental_context in experimental_contexts:
        if not experimental_context is None:
            proj.query_with_file(in_filepath = "%s/test_%s_%s.csv" % (dir_data, filename_substring, 
                                                                      experimental_context), 
                                 in_tags = {"context": experimental_context})