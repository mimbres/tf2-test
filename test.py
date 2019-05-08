#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tensorflow==2.0.0-alpha0
"""
Created on Wed May  8 03:41:25 2019

@author: mimbres
"""

import tensorflow as tf
tf.executing_eagerly()

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))


if tf.test.is_gpu_available():
  with tf.device("gpu:0"):
    v = tf.Variable(tf.random.normal([1000, 1000]))
    v = None  # v no longer takes up GPU memory