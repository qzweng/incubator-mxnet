# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()
    
    DLSCHEDULER="13.57.18.47" # MASTER
    DLSERVER="54.219.182.228" # MASTER
    DLWORKER1="13.56.159.154" # DLWORKER1
    DLWORKER2="13.57.15.78" # DLWORKER2

    import os

    os.environ.update({
      "DMLC_ROLE": "worker",
      "DMLC_PS_ROOT_URI": DLSCHEDULER,
      "DMLC_PS_ROOT_PORT": "9000",
      "DMLC_NUM_SERVER": "1",
      "DMLC_NUM_WORKER": "2",
      "PS_VERBOSE": "1"
    })

    import mxnet as mx
    import logging
    import numpy as np
    import time
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s WORKER1 %(message)s')
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s WORKER2 %(message)s')

    kv_store = mx.kv.create('dist_sync')
    
    
    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples  = 50000,
        image_shape    = '3,28,28',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 300,
        lr             = .05,
        lr_step_epochs = '200,250',
        kvstore        = 'dist_sync',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
