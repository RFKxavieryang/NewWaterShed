import math
import keras
import numpy as np
import sys
import logging
from data_tools import *

import nibabel as nib
import matplotlib

# TODO: Generate the data all at once
class DataGenerator(keras.utils.Sequence):
    "Generate the data on the fly to be used by keras"
    def __init__(self, list_IDs, batch_size=12, dim=(88,576, 576), n_channels=3,
            n_classes=2, shuffle=True, transpose_axis = (0, 1, 2)):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.cut_dim = (self.dim[0],) + tuple(math.ceil(self.dim[i] * 3 / 5) for i in [1, 2])
        self.n_slice = self.cut_dim[transpose_axis[0]]
        self.transpose_axis = tuple(transpose_axis)
        self.load_id = None
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

      tmp = []
      for i in self.indexes:
          slice_indexes = np.arange(self.n_slice)
          np.random.shuffle(slice_indexes)
          tmp += zip([i] * self.n_slice, slice_indexes)

      self.indexes = tmp

    # TODO: Make volume caching thread local
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #  dim = (self.dim[0], self.dim[2] * 3 // 5)
        # The dimension of images in the batch
        dim = tuple(self.cut_dim[i] for i in self.transpose_axis[1:])
        X = np.empty((self.batch_size,) + dim + (self.n_channels,))
        y = np.empty((self.batch_size,) + dim, dtype=int)
        y_mask = np.zeros((self.batch_size,) + dim + (1,), dtype=int)
        # Generate data
        for i, (ID, k) in enumerate(list_IDs_temp):
          # Store sample
          if self.load_id != ID:
              logging.info("opening sample {}".format(ID))

              self.x = load_nrrd('/data/public/2018_AtriaSeg/Training_Set/' + ID + '/lgemri.nrrd')
              
              self.x = regularize(self.x)
              self.x = self.x  - 127
              logging.debug("minmax {} - {}".format(self.x.min(), self.x.max()))

              self.y = load_nrrd('/data/public/2018_AtriaSeg/Training_Set/' + ID + '/laendo.nrrd') // 255
              self.load_id = ID
              if self.y.shape[1] == 640:
                  self.x = self.x[:, 32 : -32, 32 : -32]
                  self.y = self.y[:, 32 : -32, 32 : -32]

              eigth = tuple(1 * i // 5 for i in self.x.shape)

              self.x = self.x[:, eigth[1] : -eigth[1], eigth[2] : -eigth[2]]

              
              self.y = self.y[:, eigth[1] : -eigth[1], eigth[2] : -eigth[2]]
              self.x = np.transpose(self.x, self.transpose_axis)
              self.y = np.transpose(self.y, self.transpose_axis)

          x = self.x
          
          X[i, :, :, 0] = x[max(0, k - 1), :, :]
          X[i, :, :, 1] = x[k, :, :]
          X[i, :, :, 2] = x[min(self.n_slice - 1, k + 1), :, :]

          # Store class
          y[i,] = self.y[k,:,:]
          y_mask[i,:,:,0] = self.y[k,:,:]
        return X,y_mask, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) * self.n_slice / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(self.list_IDs[k[0]], k[1]) for k in indexes]

        # Generate data
        X,X1, y = self.__data_generation(list_IDs_temp)

        return [X,X1], y

