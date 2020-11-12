import json

import keras
import tensorflow as tf
from keras import backend as K

from keras.layers import Conv2D,Input,MaxPooling2D,UpSampling2D,Dropout,concatenate,ZeroPadding2D,Conv2DTranspose
from keras.utils.vis_utils import model_to_dot,plot_model
from keras.engine.base_layer import Layer
from keras import activations,initializers,regularizers,constraints
import numpy as np

class CropToInputTensor(Layer):
    def __init__(self, **kwargs):
        super(CropToInputTensor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CropToInputTensor, self).build(input_shape)

    def call(self, inputs):
        tensor = inputs[0]
        input = inputs[1]

        in_shape = tf.shape(tensor)
        out_x = in_shape[1]
        out_y = in_shape[2]

        in_shape = tf.shape(input)
        in_x = in_shape[1]
        in_y = in_shape[2]

        crop_x = in_x - out_x
        crop_y = in_y - out_y

        return input[:,
                     crop_x // 2 : in_x -(crop_x // 2 + crop_x % 2),
                     crop_y // 2 : in_y -(crop_y // 2 + crop_y % 2),
                     :]

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0],) + input_shape[0][1:3] + (input_shape[1][3],)
"""Deformable Convolutional Layer
"""

class DeformableConvLayer(Conv2D):
    """Only support "channel last" data format"""
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        '''
        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kConv2DTransposeernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True)
        '''
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        '''
        # get offset, shape [batch_size, out_h, out_w, filter_h * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filter =self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias
        '''
         
        input_shape = K.shape(inputs)
        # add padding if needed
        inputs = self._pad_input(inputs)

        #offset_shape = K.shape(offset)  
        # some length
        batch_size, in_h, in_w, channel_in = input_shape[0],input_shape[1],input_shape[2],input_shape[3]
        offset = tf.random.normal([batch_size, in_h, in_w, self.filters*18], mean = 0.0, stddev = 1.0)
        offset_shape = K.shape(offset) 
        out_h, out_w = offset_shape[1],offset_shape[2]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        #y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y = tf.reshape(y, [K.shape(y)[0], K.shape(y)[1], K.shape(y)[2], -1])
        x = tf.reshape(x, [K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], -1])
        y, x = [tf.to_float(i) for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0., tf.to_float(in_h - 1))
        x = tf.clip_by_value(x, 0., tf.to_float(in_w - 1))
        # get four coordinates of points around (x, y)
        y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        #y1, x1 = y0 + 1, x0 + 1
        # clip
        #y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        #x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]
        
        # get pixel values
        #indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        #p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i) for i in indices]
        '''
        # cast to float
        x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])
        '''
        pixels = DeformableConvLayer._get_pixel_values_at_point(inputs, [y0,x0])
       # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'SAME')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = K.shape(inputs)[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """

        feat_h, feat_w = feature_map_size[0],feature_map_size[1]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))

        x = tf.reshape(x, [1, K.shape(x)[0], K.shape(x)[1], 1])  # shape [1, h, w, 1]
        y = tf.reshape(y, [1, K.shape(y)[0], K.shape(y)[1], 1])  # shape [1, h, w, 1]
        #x, y = [tf.reshape(K.shape(i), [1, K.shape(*i), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'SAME')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = K.shape(y)[0],K.shape(y)[1],K.shape(y)[2],K.shape(y)[3]
        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)


class DeformableConvLayer_change(Layer):
    """Only support "channel last" data format"""
    def __init__(self,
                 units,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=units.
        """
        super(DeformableConvLayer_change, self).__init__(**kwargs)
        self.units=units
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.dilation_rate=dilation_rate
        self.activation=activations.get(activation)
        self.use_bias=use_bias
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.bias_initializer=initializers.get(bias_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        self.bias_regularizer=regularizers.get(bias_regularizer)
        self.activity_regularizer=regularizers.get(activity_regularizer)
        self.kernel_constraint=constraints.get(kernel_constraint)
        self.bias_constraint=constraints.get(bias_constraint)
        self.kernel = None
        self.bias = None
        if num_deformable_group is None:
            num_deformable_group = units
        if units % num_deformable_group != 0:
            raise ValueError('"units" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[0][-1])
        # kernel_shape = self.kernel_size + (input_dim, self.units)
        # we want to use depth-wise conv
        kernel_shape = (self.kernel_size[0],self.kernel_size[1]) + (input_dim, self.units)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)

        self.built = True
        #super(DeformableConvLayer_change, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h * filter_w * channel_out * 2]

        input_fp,gt_mask = inputs
        input_shape = K.shape(input_fp)
        # add padding if needed
        input_fp = self._pad_input(input_fp)
        gt_mask = self._pad_input(gt_mask)

        # some length
        filter_h, filter_w = self.kernel_size
        batch_size, in_h, in_w, channel_in = input_shape[0],input_shape[1],input_shape[2],input_shape[3]
        gt_mask = tf.tile(gt_mask, [batch_size, 1, 1, input_shape[3]])  
        offset = tf.random.normal([batch_size, in_h, in_w, self.units*filter_h * filter_w * 2], mean = 0.0, stddev = 1.0)
        offset_shape = K.shape(offset) 
        out_h, out_w = offset_shape[1],offset_shape[2]  # output feature map size


        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]


        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        #y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y = tf.reshape(y, [K.shape(y)[0], K.shape(y)[1], K.shape(y)[2], -1])
        x = tf.reshape(x, [K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], -1])
        y, x = [tf.to_float(i) for i in [y, x]]

        p_mask = DeformableConvLayer_change._get_pixel_values_at_point(gt_mask, [tf.to_int32(tf.floor(y)), tf.to_int32(tf.floor(x))])
        p_mask_offset = DeformableConvLayer_change._get_pixel_values_at_point(gt_mask, [tf.to_int32(tf.floor(y+ y_off)), tf.to_int32(tf.floor(x+ x_off))])
        y_off = y_off*tf.to_float(p_mask != p_mask_offset)
        x_off = x_off*tf.to_float(p_mask != p_mask_offset)
        # add offset
        y, x = y + y_off, x + x_off

        y = tf.clip_by_value(y, 0., tf.to_float(in_h - 1))
        x = tf.clip_by_value(x, 0., tf.to_float(in_w - 1))
        # get four coordinates of points around (x, y)
        y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        '''
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer_change._get_pixel_values_at_point(input_fp, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])
        '''
        pixels = DeformableConvLayer_change._get_pixel_values_at_point(input_fp, [y0,x0]) 
        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.units // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'SAME')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, -1])
        #out = tf.reduce_sum(out, axis=-1)

        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = K.shape(inputs)[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """

        feat_h, feat_w = feature_map_size[0],feature_map_size[1]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))

        x = tf.reshape(x, [1, K.shape(x)[0], K.shape(x)[1], 1])  # shape [1, h, w, 1]
        y = tf.reshape(y, [1, K.shape(y)[0], K.shape(y)[1], 1])  # shape [1, h, w, 1]
        #x, y = [tf.reshape(K.shape(i), [1, K.shape(*i), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'SAME')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = K.shape(y)[0],K.shape(y)[1],K.shape(y)[2],K.shape(y)[3]
        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[0][1:3] + (self.units,)



def VGG16():
    inputs = keras.layers.Input(shape=(346,346,3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    model = keras.Model(inputs, pool5, name='vgg16')

    return model

def VGG16_deform():
    inputs = keras.layers.Input(shape=(346,346,3))
    conv1 = DeformableConvLayer(16, [3, 3], num_deformable_group=None, activation='relu')(inputs)
    #conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    conv2 = DeformableConvLayer(32, [3, 3], num_deformable_group=None, activation='relu')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    #conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    conv3 = DeformableConvLayer(64, [3, 3], num_deformable_group=None, activation='relu')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    #conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    model = keras.Model([inputs], pool5, name='vgg16')

    return model
def VGG16_deform_change_small_data():
    inputs = keras.layers.Input(shape=(346,346,3), name='input1')
    #inputs_gt = keras.layers.Input(shape=(346,346,1), name='input2')
    conv1 = DeformableConvLayer(16, [3, 3], num_deformable_group=None, activation='relu')(inputs)
    #conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    #inputs_gt_1 = MaxPooling2D((2, 2), strides=(2, 2))(inputs_gt)
    conv2 = DeformableConvLayer(32, [3, 3], num_deformable_group=None, activation='relu')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same',name='conv2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    #inputs_gt_2 = MaxPooling2D((2, 2), strides=(2, 2))(inputs_gt_1)
    conv3 = DeformableConvLayer(64, [3, 3], num_deformable_group=None, activation='relu')(pool2)
    #conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same',name='conv3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv4')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv5')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    #model = keras.Model([inputs,inputs_gt], pool5, name='vgg16')

    
    conv1 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv3)
    conv4 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv4)
    conv5 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv5)

    upsampled_layers2 = Conv2DTranspose(filters=16, kernel_size=4 ,strides=2)(conv2)
    upsampled_layers3 = Conv2DTranspose(filters=16, kernel_size=8 ,strides=4)(conv3)
    upsampled_layers4 = Conv2DTranspose(filters=16, kernel_size=16 ,strides=8)(conv4)
    upsampled_layers5 = Conv2DTranspose(filters=16, kernel_size=32 ,strides=16)(conv5)

    cropped_layers2 = CropToInputTensor()([conv1, upsampled_layers2])
    cropped_layers3 = CropToInputTensor()([conv1, upsampled_layers3])
    cropped_layers4 = CropToInputTensor()([conv1, upsampled_layers4])
    cropped_layers5 = CropToInputTensor()([conv1, upsampled_layers5])
    concatenated = keras.layers.Concatenate()([conv1,cropped_layers2,cropped_layers3,cropped_layers4,cropped_layers5])
    x = keras.layers.Dropout(0.7)(concatenated)
    output = Conv2D(2, (1,1), activation='softmax', kernel_initializer=keras.initializers.Constant(0.01))(x)
    
    model = keras.Model(inputs=[inputs], outputs=output)
    #model.summary()
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model

def VGG16_deform_change():
    inputs = keras.layers.Input(shape=(346,346,3), name='input1')
    inputs_gt = keras.layers.Input(shape=(346,346,1), name='input2')
    conv1 = DeformableConvLayer_change(64, [3, 3], num_deformable_group=None, activation='relu')([inputs,inputs_gt])
    #conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

    # Block 2
    inputs_gt_1 = MaxPooling2D((2, 2), strides=(2, 2))(inputs_gt)
    conv2 = DeformableConvLayer_change(128, [3, 3], num_deformable_group=None, activation='relu')([pool1,inputs_gt_1])
    #conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',name='conv2')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

    # Block 3
    #inputs_gt_2 = MaxPooling2D((2, 2), strides=(2, 2))(inputs_gt_1)
    #conv3 = DeformableConvLayer_change(256, [3, 3], num_deformable_group=None, activation='relu')([pool2,inputs_gt_2])
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same',name='conv3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3)

    # Block 4

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same',name='conv4')(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    # Block 5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',name='conv5')(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)
    #model = keras.Model([inputs,inputs_gt], pool5, name='vgg16')

    '''
    conv1 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv3)
    conv4 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv4)
    conv5 = Conv2D(filters=16, kernel_size=(3,3), padding="same")(conv5)

    upsampled_layers2 = Conv2DTranspose(filters=16, kernel_size=4 ,strides=2)(conv2)
    upsampled_layers3 = Conv2DTranspose(filters=16, kernel_size=8 ,strides=4)(conv3)
    upsampled_layers4 = Conv2DTranspose(filters=16, kernel_size=16 ,strides=8)(conv4)
    upsampled_layers5 = Conv2DTranspose(filters=16, kernel_size=32 ,strides=16)(conv5)

    cropped_layers2 = CropToInputTensor()([conv1, upsampled_layers2])
    cropped_layers3 = CropToInputTensor()([conv1, upsampled_layers3])
    cropped_layers4 = CropToInputTensor()([conv1, upsampled_layers4])
    cropped_layers5 = CropToInputTensor()([conv1, upsampled_layers5])
    concatenated = keras.layers.Concatenate()([conv1,cropped_layers2,cropped_layers3,cropped_layers4,cropped_layers5])
    x = keras.layers.Dropout(0.7)(concatenated)
    output = Conv2D(2, (1,1), activation='softmax', kernel_initializer=keras.initializers.Constant(0.01))(x)
    '''
    model = keras.Model(inputs=[inputs,inputs_gt], outputs=pool5)
    #model.summary()
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


def deform_net_original():

    vgg16 = VGG16_deform()
    
    
    layers = [l for l in vgg16.layers]
    # Create the specialized layers
    specialized_layers = []
     
    level = 0
    for i, layer in enumerate(layers):
        if type(layer) != keras.layers.pooling.MaxPooling2D:
            continue
        level += 1
        previous = layers[i - 1]
        new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(previous.output)
        specialized_layers.append(new_layer)
    

    upsampled_layers = [keras.layers.Conv2DTranspose(filters=8, kernel_size=2 * (2 ** i),
                                                     strides=2 ** i
                                                     )(layer)  if i != 0 else layer
            for i, layer in enumerate(specialized_layers)]

    cropped_layers = [CropToInputTensor()([layers[0].output, layer])
                          for i, layer in enumerate(upsampled_layers)]

    concatenated = keras.layers.Concatenate()(cropped_layers)
    x = keras.layers.Dropout(0.7)(concatenated)
    output = Conv2D(2, (1,1), activation='softmax', kernel_initializer=keras.initializers.Constant(0.01))(x)

    model = keras.Model(inputs=[layers[0].input], outputs=output)
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model
def deform_net(vgg_weights_filename):

    #vgg16 = VGG16()
    #from keras.applications.vgg16 import VGG16
    #vgg16 = VGG16(weights=vgg_weights_filename, include_top=False)
    vgg16 = VGG16_deform_change()
    vgg16.load_weights(vgg_weights_filename)
    
    layers = [l for l in vgg16.layers]
    # Create the specialized layers
    specialized_layers = []
    ''' 
    level = 0
    for i, layer in enumerate(layers):
        if type(layer) != keras.layers.pooling.MaxPooling2D:
            continue
        level += 1
        previous = layers[i - 1]
        new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(previous.output)
        specialized_layers.append(new_layer)
    '''
    new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(vgg16.get_layer('conv1').output)
    specialized_layers.append(new_layer)
    new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(vgg16.get_layer('conv2').output)
    specialized_layers.append(new_layer)
    new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(vgg16.get_layer('conv3').output)
    specialized_layers.append(new_layer)
    new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(vgg16.get_layer('conv4').output)
    specialized_layers.append(new_layer)
    new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(vgg16.get_layer('conv5').output)
    specialized_layers.append(new_layer)

    upsampled_layers = [keras.layers.Conv2DTranspose(filters=16, kernel_size=2 * (2 ** i),
                                                     strides=2 ** i
                                                     )(layer)  if i != 0 else layer
            for i, layer in enumerate(specialized_layers)]

    cropped_layers = [CropToInputTensor()([layers[0].output, layer])
                          for i, layer in enumerate(upsampled_layers)]

    concatenated = keras.layers.Concatenate()(cropped_layers)
    x = keras.layers.Dropout(0.7)(concatenated)
    output = Conv2D(2, (1,1), activation='softmax', kernel_initializer=keras.initializers.Constant(0.01))(x)

    model = keras.Model(inputs=[layers[0].input,layers[1].input], outputs=output)
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def create_model(vgg_weights_filename):
    from keras.applications.vgg16 import VGG16
    vgg16 = VGG16(weights=vgg_weights_filename, include_top=False)
    layers = [l for l in vgg16.layers]

    # Create the specialized layers
    specialized_layers = []
    level = 0
    for i, layer in enumerate(layers):
        if type(layer) != keras.layers.pooling.MaxPooling2D:
            continue
        level += 1
        previous = layers[i - 1]
        new_layer = Conv2D(filters=16, kernel_size=(3,3), padding="same")(previous.output)
        specialized_layers.append(new_layer)

    upsampled_layers = [keras.layers.Conv2DTranspose(filters=16, kernel_size=2 * (2 ** i),
                                                     strides=2 ** i
                                                     )(layer)  if i != 0 else layer
            for i, layer in enumerate(specialized_layers)]

    cropped_layers = [CropToInputTensor()([layers[0].output, layer])
                          for i, layer in enumerate(upsampled_layers)]

    location = keras.layers.Input(shape=(1,))

    concatenated = keras.layers.Concatenate()(cropped_layers)
    x = keras.layers.Dropout(0.7)(concatenated)
    output = Conv2D(2, (1,1), activation='softmax', kernel_initializer=keras.initializers.Constant(0.01))(x)

    model = keras.Model(inputs=layers[0].input, outputs=output)
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def load_model(weights_file):
    with open(weights_file.replace('.h5', '.json')) as json_file:
        model_json = '\n'.join(json_file.readlines())
    model = keras.models.model_from_json(model_json, custom_objects={'CropToInputTensor': CropToInputTensor, 'DeformableConvLayer': DeformableConvLayer, 'DeformableConvLayer_change': DeformableConvLayer_change,'tf': tf,'keras':keras})
    model.load_weights(weights_file)
    return model

def load_model_json(weights_file):
    with open(weights_file.replace('.h5', '.json')) as json_file:
       return json.load(json_file)
