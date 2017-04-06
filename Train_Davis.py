from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys, os

import Davis_FCN
import read_davis
import scipy.misc

from config import *

def main(argv):
    
    # read current file index
    file_index = 0
    if os.path.exists('./file_index'):
        f = open('./file_index')
        for line in f:
            file_index = int(line)
        f.close()
    f = open('./file_index', 'a+')

    # import data
    davis_reader = read_davis.DavisReader(file_index)
    
    # Create the model
    fcn = Davis_FCN.FCN() 
    x = tf.placeholder(tf.float32)#, shape=[BATCH_SIZE, None, None, 4]) 
    y_ = tf.placeholder(tf.float32)#, shape=[BATCH_SIZE, None, None, NUM_CLASSES])
    y = fcn.build(x, train=True, num_classes=NUM_CLASSES, 
                random_init_fc8=True, debug=True)

	# imbalanced weighted
    pixels = tf.reduce_sum(y_, [1,2])
    pixels_total = tf.expand_dims(tf.reduce_sum(pixels,1),1)
    ratio = tf.divide(pixels, pixels_total)
    ratio = tf.expand_dims(ratio, 1)
    ratio = tf.expand_dims(ratio, 1)
    y_weighted = tf.multiply(y_, ratio)

	# Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_weighted, logits=y))
    train_step = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    train_step = train_step.minimize(cross_entropy)

    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    saver = tf.train.Saver()
    if not os.path.exists('models'):
         os.makedirs('models')
    else:
        saver.restore(sess, "./models/model%s"%str(MODEL_INDEX-1))
        print("Model%s restored ..."%str(MODEL_INDEX-1))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Training
    print('='*40)
    print('Training ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys = davis_reader.next_batch()
        entropy = 0
        _, loss_val = sess.run([train_step,cross_entropy], 
                                feed_dict={x: batch_xs, y_: batch_ys})
        loss.append(loss_val)
        if i%100==0:
            save_path = saver.save(sess, "./models/model%s"%MODEL_INDEX)      
            f.write(str(file_index+i+1)+'\n')
        log = 'Iteration: %s'%str(i) + \
                ' | Model saved in file: %s'%save_path + ' | Cross entropy loss: %s'%str(loss_val) 
        print(log)
    f.close()
    np.save('./models/trCrossEntropyLoss%s'%MODEL_INDEX, np.array(loss))


if __name__=='__main__':
    tf.app.run(main=main, argv=[])



