#!/usr/bin/python3

import tensorflow as tf;

class MNIST(object):
  def __init__(self,):
    (self.train_x, _), (self.test_x, _) = tf.keras.datasets.mnist.load_data();
  def load_trainset(self,):
    return tf.data.Dataset.from_tensor_slices(self.train_x);
  def load_testset(self,):
    return tf.data.Dataset.from_tensor_slices(self.test_x);
