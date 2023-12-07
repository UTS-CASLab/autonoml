# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:33:41 2023

@author: David J. Kedziora
"""

import autonoml as aml       

dir_data = "./data/iris"

if __name__ == '__main__':

    proj = aml.AutonoMachine(do_mp = False)

    proj.ingest_file(in_filepath = "%s/train_iris.csv" % dir_data)
    proj.query_with_file(in_filepath = "%s/test_iris.csv" % dir_data)

    proj.info_storage()

    proj.learn("variety", in_strategy = aml.import_strategy("./test_proj_iris.strat"))