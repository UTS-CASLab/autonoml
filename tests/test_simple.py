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
    proj.ingest_file(in_filepath = "%s/train_dummy.csv" % dir_data, in_tags = {"file": 2})
    proj.ingest_file(in_filepath = "%s/train_dummy.csv" % dir_data, in_tags = {"file": 3})
    proj.query_with_file(in_filepath = "%s/test_dummy.csv" % dir_data)

    proj.info_storage()

    proj.learn("target_int", in_strategy = aml.import_strategy("./test_simple.strat"), 
               in_keys_features = ["file"], do_exclude = True,
               in_keys_allocation = [("file", aml.AllocationMethod.LEAVE_ONE_OUT)])

    # if not True:
    #     # Run the following commands in online mode after data loading is done.
    #     o = proj.data_storage.observations
    #     q = proj.data_storage.queries
    #     print(o.data)
    #     print(q.data)
    #     o1, o2 = o.split_by_fraction(0.25)
    #     print(o1.data)
    #     print(o2.data)