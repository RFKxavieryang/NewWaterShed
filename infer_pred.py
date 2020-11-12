# First setup tensorflow and keras, import the rest after
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

import json
import sys
import keras
import keras.models
import progress.bar
import nibabel as nib
import metrics
import timeit
from data_tools import *
import model
import os
import matplotlib
import imgaug.augmenters as iaa	

from timeit import default_timer as timer

#Parse the arguments first
def parse_args():
    from docopt import docopt
    from ast import literal_eval

    from schema import Schema, Use, And, Or

    usage = """
Usage: infer.py -h
       infer.py [--training_set=<path>] <model_path> <output_location>

Options:
    -h --help                  show this
    --training_set=<path>      the training dataset location
    <model_path>               the path to the model weights
    <output_location>          the path to the output folder

    """

    args = docopt(usage)
    schema = Schema({
        '--help': False,
        '--training_set': Or(None, And(Use(str), os.path.isdir)),
        '<output_location>': str,
        '<model_path>': str,
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
#args = parse_args()

with open("list_id.json", "r") as f:
    list_id = json.load(f)

training_folder ='/data/public/2018_AtriaSeg/Training_Set/'
if training_folder is None:
    training_folder = '/data/public/2018_AtriaSeg/Training_Set/'

'''
#model = model.create_model(weights_file)
model_filename ='/lrde/home/zz/Heart/atriaseg2018/zz_whu/model_epoch_4.h5'
model_json = model.load_model_json(model_filename)
model = model.load_model(model_filename)
optimizer = keras.optimizers.Adam(epsilon=0.002, amsgrad=True)
model.compile(optimizer, loss='categorical_crossentropy',
    metrics=['accuracy', metrics.dice_coef, metrics.dice_coef_no_bg])

if not os.path.exists('/lrde/home/zz/Heart/atriaseg2018/zz_whu/result'):
    os.makedirs('/lrde/home/zz/Heart/atriaseg2018/zz_whu/result')
'''

model_filename ='/home/mio/zz_heart/model_epoch_8.h5'
model_json = model.load_model_json(model_filename)
model = model.deform_net('/data/public/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
#model = keras.utils.multi_gpu_model(model,1)
model.load_weights(model_filename)

i = 0
full_time = 0
noload_time = 0
predtime = 0
for ID in list_id[20:40]:
    #Y = np.zeros((88,346,346,3),np.float)
    axis = '012_'
    dataset = '20_40'
    print(ID, i)
    sload = timer()
    base = training_folder+ID+'/'
    mri = load_nrrd('/data/public/2018_AtriaSeg/Training_Set/' + ID + '/lgemri.nrrd')
    gt = load_nrrd('/data/public/2018_AtriaSeg/Training_Set/' + ID + '/laendo.nrrd') // 255
    start = timer()
    norm = regularize(mri)

    
    #aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    aug = iaa.SaltAndPepper(0.1)
    #aug = iaa.Cutout(fill_mode="gaussian")
    for j in range(norm.shape[0]):
      norm[j,:,:] = aug.augment_image(norm[j,:,:])
    '''
    img_noise = np.transpose(norm, tuple((1, 2, 0)))
    img_noise = nib.Nifti1Image(img_noise.astype('float64'), np.eye(4))
    directory = "result_segmentation/result_{}/{}/image_noise".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_noise.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/zz_whu/result_segmentation/result_{}/{}/image_noise'.format(axis,dataset), '{}.nii'.format(ID)))
    '''

    norm -= 127 
    
    if norm.shape[1] == 640:
      norm = norm[:, 32 : -32, 32 : -32]
      gt = gt[:, 32 : -32, 32 : -32]

    eigth = tuple(1 * i // 5 for i in norm.shape)
    norm = norm[:, eigth[1] : -eigth[1], eigth[2] : -eigth[2]]
    gt = gt[:, eigth[1] : -eigth[1], eigth[2] : -eigth[2]]
    
    if "transpose_axis" in model_json:
        norm = np.transpose(norm, model_json["transpose_axis"])
        gt = np.transpose(gt, model_json["transpose_axis"])
    else:
        #transpose axis
        #  norm = np.transpose(norm, (2, 0, 1))
        #  gt = np.transpose(gt, (2, 0, 1))
        pass

    X = np.zeros(norm.shape + (3,))
    gt_mask = np.zeros(norm.shape+(1,))
    for k in range(norm.shape[0]):
        X[k, :, :, 0] = norm[max(0, k - 1), :, :]
        X[k, :, :, 1] = norm[k, :, :]
        X[k, :, :, 2] = norm[min(norm.shape[0] - 1, k + 1), :, :]
        gt_mask[k, :, :, 0] = gt[k, :, :]

    sp = timer()
    pred = model.predict([X,gt_mask], batch_size = 8, verbose=False)
    ep = timer()


    if "transpose_axis" in model_json:
        inverted_transpose = np.argsort(model_json["transpose_axis"])
        norm = np.transpose(norm, inverted_transpose)
        gt = np.transpose(gt, inverted_transpose)
        pred = np.transpose(pred, tuple(inverted_transpose) + (3,))

    
    proba = pred
    pred = np.argmax(pred, 3)
    pred = np.transpose(pred, tuple((1, 2, 0)))
    gt = np.transpose(gt, tuple((1, 2, 0)))

    end = timer()
    full_time += end - sload
    noload_time += end - start
    predtime += ep -sp
    pred = pred.astype('float64')
    img = nib.Nifti1Image(pred, np.eye(4))
    directory = "result_segmentation/result_{}/{}/pre".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/home/mio/zz_heart/result_segmentation/result_{}/{}/pre'.format(axis,dataset), '{}.nii'.format(ID)))
    img = nib.Nifti1Image(gt.astype('float64'), np.eye(4))
    directory = "result_segmentation/result_{}/{}/gt".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/home/mio/zz_heart/result_segmentation/result_{}/{}/gt'.format(axis,dataset), '{}_gt.nii'.format(ID)))
    i += 1
    
print(full_time / 20)
print(noload_time / 20)
print(predtime / 20)
    
