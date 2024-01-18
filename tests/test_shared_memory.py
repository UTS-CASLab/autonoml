# -*- coding: utf-8 -*-
"""
This script tests writing to and reading from memory-mapped files.

Created on Fri Oct 27 17:06:45 2023

@author: David J. Kedziora
"""

from autonoml.data_storage import DataCollection, SharedMemoryManager
from autonoml.solver_ops import ProcessInformation, prepare_data

import numpy as np
import pyarrow as pa

n_rows = 2000
n_cols = 500

if __name__ == '__main__':

    # Create data.
    cols = list()
    for _ in range(n_cols):
        col = np.random.rand(n_rows).tolist()
        cols.append(col)

    cols.append(np.random.rand(n_rows).tolist())

    keys_features = ["col_%i" % i for i in range(n_cols)]
    key_target = "target"

    # Convert to PyArrow table.
    data = pa.Table.from_arrays(cols, names = keys_features + [key_target])
    collection = DataCollection(in_data = data)

    # Break up into sets.
    info_process = ProcessInformation(in_keys_features = keys_features, in_key_target = key_target)
    output = prepare_data(in_collection = collection, in_info_process = info_process,
                          in_frac_validation = 0.25, in_n_sets = 3)
    observations, sets_training, sets_validation = output

    # Create memory-mapped files and read back from them.
    manager = SharedMemoryManager()
    manager.save_observations(observations, sets_training, sets_validation)
    observations_read, sets_training_read, sets_validation_read = manager.load_observations()

    # Check whether the original tables/arrays are equivalent to the ones read in.
    print("Observations (x) equal: %s" % (observations.x.equals(observations_read.x)))
    print("Observations (y) equal: %s" % (observations.y.equals(observations_read.y)))
    idx_sets = 0
    for set_training, set_training_read, set_validation, set_validation_read in zip(
        sets_training, sets_training_read, sets_validation, sets_validation_read):

        print("Training sets %i (x) equal: %s" % (idx_sets, set_training.x.equals(set_training_read.x)))
        print("Training sets %i (y) equal: %s" % (idx_sets, set_training.y.equals(set_training_read.y)))
        print("Validation sets %i (x) equal: %s" % (idx_sets, set_validation.x.equals(set_validation_read.x)))
        print("Validation sets %i (y) equal: %s" % (idx_sets, set_validation.y.equals(set_validation_read.y)))
        idx_sets += 1

    # # Delete references to memory-mapped files so that the files can be cleaned.
    # del observations_read, sets_training_read, sets_validation_read
    # del set_training, set_training_read, set_validation, set_validation_read
    # manager.decrement_uses()