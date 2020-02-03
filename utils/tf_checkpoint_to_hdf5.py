#!/usr/bin/env python3

import argparse
import os
import re
import sys

import h5py
import tensorflow as tf

parser = argparse.ArgumentParser(
    description='Convert Tensorflow checkpoint to HDF5.')
parser.add_argument(
    'checkpoint',
    metavar='CHECKPOINT',
    help='The checkpoint base path')
parser.add_argument('hdf5', metavar='HDF5', help='HDF5 output')

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    model_vars = tf.train.list_variables(checkpoint_path)

    with h5py.File(args.hdf5, "w") as hdf5:
        ignore = re.compile("adam_v|adam_m|global_step|cls|pooler")
        for var in model_vars:
            (var, shape) = var

            # Skip unneeded layers
            if ignore.search(var):
                continue

            # Rewrite some variable names
            renamedVar = var.replace("kernel", "weight")
            renamedVar = renamedVar.replace("gamma", "weight")
            renamedVar = renamedVar.replace("beta", "bias")

            print("Adding %s..." % renamedVar, file=sys.stderr)

            # Retrieve the tensor associated with the variable
            # and store it in the HDF5 file.
            tensor = tf.train.load_variable(checkpoint_path, var)
            hdf5.create_dataset(renamedVar, data=tensor)
