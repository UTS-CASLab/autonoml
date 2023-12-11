# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:03:05 2023

@author: David J. Kedziora
"""

import autonoml as aml       

dir_data = "./data/dummy"

if __name__ == '__main__':

    proj = aml.AutonoMachine(do_mp = False)

    proj.ingest_file(in_filepath = "%s/train_dummy.csv" % dir_data, in_tags = {"file": 1})
    proj.ingest_file(in_filepath = "%s/train_dummy.csv" % dir_data, in_tags = {"file": 1, "special": 5})
    proj.ingest_file(in_filepath = "%s/train_dummy.csv" % dir_data, in_tags = {"file": 2})
    proj.query_with_file(in_filepath = "%s/test_dummy.csv" % dir_data)
    proj.query_with_file(in_filepath = "%s/test_dummy.csv" % dir_data, in_tags = {"file": 2})
    proj.query_with_file(in_filepath = "%s/test_dummy.csv" % dir_data, in_tags = {"alt": "a", "file": 3})

    # proj.info_storage()

    proj.learn("target_int", do_immediate_responses = False,
               in_strategy = aml.import_strategy("./test_simple.strat"),
               in_tags_allocation = [("file", aml.AllocationMethod.LEAVE_ONE_OUT)])