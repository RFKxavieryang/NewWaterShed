from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=0.1):
    
    y_true_f = tf.cast(K.flatten(tf.argmax(y_true, -1)), tf.float64)
    y_pred_f = tf.cast(K.flatten(tf.argmax(y_pred, -1)), tf.float64)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_no_bg(y_true, y_pred, smooth=1):
    c1 = tf.greater(y_true[..., 1], 0.5)
    c2 = tf.greater(y_pred[..., 1], 0.5)
    y_true_f = K.flatten(tf.boolean_mask(y_true[..., 1], tf.logical_or(c1, c2)))
    y_pred_f = K.flatten(tf.boolean_mask(y_pred[..., 1], tf.logical_or(c1, c2)))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


