#!/usr/bin/python3

import tensorflow as tf;

class MNIST(object):
  def __init__(self,):
    (self.train_x, _), (self.test_x, _) = tf.keras.datasets.mnist.load_data();
  def sampler_generator(self, x):
    def sampler():
      for sample in x:
        yield sample;
      return;
    return sampler;
  def load_trainset(self,):
    return tf.data.Dataset.from_generator(self.sampler_generator(self.train_x), (tf.uint8,), (tf.TensorShape([28, 28])));
  def load_testset(self,):
    return tf.data.Dataset.from_generator(self.sampler_generator(self.test_x), (tf.uint8,), (tf.TensorShape([28, 28])));
