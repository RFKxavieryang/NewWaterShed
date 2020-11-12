#!/usr/bin/env python3


#Parse the arguments first
def parse_args():
    from docopt import docopt
    from ast import literal_eval

    from schema import Schema, Use, And, Or
    import os

    usage = """
Usage: train.py [-h] [-t <axis>] [-b <bs>]
                [-j jobs] [-e <epochs>]
                [--initial_epoch=<ie>]
                [--training_set=<path>]
                [--weights_file=<path>]
                [--model_name=<name>]

Options:
    -h --help                  show this
    -t --transpose=<axis>      the transpose tuple to use [default: (0, 1, 2)]
    -b --batch_size=<bs>       the batch size to use [default: 2]
    -j --jobs=<jobs>           the number of jobs to use [default: 16]
    -i --initial_epoch=<ie>    the initial epoch, enable retraining if nonzero [default: 0]
    -e --epochs=<epochs>       the number of epochs to run [default: 1]
    --training_set=<path>      the training dataset location
    --weights_file=<path>      the path to VGG16 weights or to the inital model weights
    --model_name=<name>        the name of the model [default: model]
    """

    args = docopt(usage)
    schema = Schema({
        '--help': False,
        '--transpose': And(Use(literal_eval), (0, 1, 2), lambda t: len(t) == 3),
        '--batch_size': Use(int),
        '--jobs': Use(int),
        '--initial_epoch': Use(int),
        '--epochs': Use(int),
        '--training_set': Or(None, And(Use(str), os.path.isdir)),
        '--weights_file': Or(None, And(Use(str), os.path.exists)),
        '--model_name': str
        })
    try:
        args = schema.validate(args)
    except Exception as e:
        print(args)
        print(e)
        import sys
        sys.exit(1)

    #args['--transpose'] = literal_eval(args['--transpose'])

    return args

args = parse_args()

# Setup tensorflow and keras, import the rest after
import os
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Make sure we are not using all GPUs

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True # Try not to eat all the GPU memory
sess = tf.Session(config=config)

import keras
from keras import backend as K
K.set_session(sess)

from keras.layers import Conv2D
from keras.optimizers import SGD
from keras.utils.vis_utils import model_to_dot
import h5py
import itertools
import json
import logging
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys

from model import create_model, load_model,deform_net,VGG16_deform_change,deform_net_original,VGG16_deform_change_small_data
import metrics
from metrics import dice_coef
from data import DataGenerator

from multi_gpu import ParallelModel

def list_avaible_tf_devices():
    '''Utility function, mainly used to check the correct GPU is in use.
    CUDA_VISIBLE_DEVICES may not use the same numbering scheme than nvidia-smi for example'''
    from tensorflow.python.client import device_lib
    return [device.name for device in device_lib.list_local_devices()]


def get_logdir():
    "Get the current log directory (ie. ./logs/runX)"
    # TODO: Use timestamps instead of the date
    base = "/home/mio/zz_heart/log/run"
    for i in itertools.count():
        if not os.path.exists(base + str(i)):
            break

    return base + str(i)


def save_model(model, filename='model.json', epoch=None):
    "Save the model as json, and its weights in an h5 file"
    # TODO: make something more sane
    model_json = json.loads(model.to_json())
    print(args)
    model_json["transpose_axis"] = args["--transpose"]
    model_json["epoch"] = epoch
    model_json["training_args"] = args
    with open(filename, 'w') as f:
        json.dump(model_json, f)
    #model.save_weights(os.environ.get('WEIGHT_FILEPATH', 'model_weights.h5'))
    model.save_weights(filename.replace('json', 'h5'))

    with open(LOGDIR+'/' + filename, 'w') as f:
        json.dump(model_json, f)
    model.save_weights(LOGDIR+'/' + filename.replace('json', 'h5'))


class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, modelname='model'):
        self.modelname = modelname

    def on_epoch_end(self, epoch, logs=None):
        save_model(self.model, filename=self.modelname + '_epoch_' + str(epoch) + '.json', epoch=epoch)

# The path to the challenge dataset
training_folder = args['--training_set'] if args['--training_set'] \
             else '/data/public/2018_AtriaSeg/Training Set/'

# The path to VGG16 weight file
weights_file = args['--weights_file'] if args['--weights_file'] \
             else '/data/public/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

LOGDIR=get_logdir()
os.mkdir(LOGDIR)

print(list_avaible_tf_devices())

# Open the list of sample IDs
with open("list_id.json", "r") as f:
    list_id = json.load(f)

training_generator = DataGenerator(list_id[:20]+list_id[-60:],
                                   transpose_axis=args['--transpose'],
                                   batch_size=args['--batch_size']) # Make sure these don't overlap


validation_generator = DataGenerator(list_id[20:40],
                                     transpose_axis = args['--transpose'],
                                     batch_size=args['--batch_size'])


class FrequentTensorBoard(keras.callbacks.TensorBoard):
    "This should log more frequently than the default keras callback (once per batch)"
    def __init__(self, log_dir='./logs',
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None):
        super().__init__()

    def on_epoch_end(self, epoch, logs=()):
        pass

    def on_batch_end(self, epoch, logs=()):
        super().on_epoch_end(epoch, logs)


# Setup debug logging
logging.basicConfig(filename='train.log', level=logging.DEBUG, filemode='w')

tb_callback = FrequentTensorBoard(log_dir=LOGDIR)

if args['--initial_epoch'] == 0:
    #model = create_model(weights_file)
    model = deform_net(weights_file)
    #model = deform_net_original()
    #model = VGG16_deform_change()
    #model = VGG16_deform_change_small_data()
    #model = ParallelModel(model, 3)
    #model =keras.utils.multi_gpu_model(model, 4)
else:
    model = load_model(weights_file)

# Print the model to dot (useful to check everything is correct)
with open('model.dot', 'w') as f:
   print(model_to_dot(model, show_shapes=True), file=f)

save_model(model, args['--model_name'] + '.json')
#model.load_weights('model_epoch_9.h5')
optimizer = keras.optimizers.Adam(lr=0.0002, amsgrad=True)
model.compile(optimizer, loss='categorical_crossentropy',
        metrics=['accuracy', metrics.dice_coef_no_bg])

# The actual training
model.fit_generator(generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=True,
        workers=args['--jobs'],
        initial_epoch=args['--initial_epoch'],
        epochs=args['--epochs'],
        callbacks=[tb_callback, SaveModelCallback(args['--model_name'])])

save_model(model, args['--model_name'] + '.json')
