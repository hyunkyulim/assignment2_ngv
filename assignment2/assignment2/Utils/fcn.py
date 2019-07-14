# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from Utils.vgg import Vgg16


def _conv_1x1(input_layer, n_classes):
    return slim.conv2d(input_layer, n_classes, [1, 1], activation_fn=None)

def _upsampling(input_layer, n_classes, ratio=2):
    return slim.conv2d_transpose(input_layer,
                                 n_classes,
                                 [ratio*2, ratio*2],
                                 [ratio, ratio],
                                 activation_fn=None)

class FcnModel(object):
    def __init__(self, input_tensor, y_true_tensor, is_training, n_classes=2):
        self._vgg = Vgg16(input_tensor, is_training)
        self._is_training = is_training
        self._n_classes = n_classes
        
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        '''Layers name in VGG: self._vgg.conv1_1, self._vgg.conv1_2, self._vgg.pool1, ..., described in ./vgg.py'''
        '''Calculate FCN8s using VGG layers and function _conv_1x1(), _upsampling()'''
        
        
        # FCN8s: (H, W, num_classes)
        FCN8s = ##

        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        self.inference_op = FCN8s
        
        logits = tf.reshape(self.inference_op, (-1, self._n_classes))
        class_labels = tf.reshape(y_true_tensor, (-1, self._n_classes))
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = class_labels)
        self.loss = cross_entropy_loss = tf.reduce_mean(cross_entropy)

        
        logits = tf.reshape(self.inference_op, (-1, self._n_classes))
        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, 1)
        
        class_labels = tf.reshape(y_true_tensor, (-1, self._n_classes))
        labels = tf.argmax(class_labels, 1)
        is_correct = tf.equal(pred_labels, labels)
        accuracy_op = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        mean_iou_op, update_op = tf.metrics.mean_iou(labels, pred_labels, self._n_classes)        
        
        self.accuracy, self.iou, self.update = accuracy_op, mean_iou_op, update_op
        
        # Summary
        
        with tf.name_scope('train_summary'):
            summary_loss = tf.summary.scalar('loss', self.loss)
            summary_acc = tf.summary.scalar('pixelwise_accuracy', self.accuracy)
            
            summary = tf.summary.merge([summary_loss, summary_acc], name='train_summary')        
        self.summary = summary