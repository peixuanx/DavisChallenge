from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys, os

import Davis
import Read_Davis
import scipy.misc

from config import *

def main(argv):
	
    # import data
    # pascal_reader = read_pascal.PascalReader()

    # Create the model
    fcn = Davis.FCN16VGG()
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y_ = tf.placeholder(tf.float32)
    y = fcn.build(x, train=True, num_classes=NUM_CLASSES, 
                random_init_fc8=True, debug=True)

    # Define prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    
    # Session Define
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, "./models/model%s.ckpt"%MODEL_INDEX)
    print("Model restored ...")
    
    # Testing
    print('='*40)
    print('Tresting ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys, filename = pascal_reader.next_test()
        err, truth, pred = sess.run([correct_prediction, tf.argmax(y_,3), tf.argmax(y,3)], 
                            feed_dict={x: batch_xs, y_: batch_ys})
        h,w = np.shape(err[0])
        loss_val = len(np.where(err[0]==False)[0])/(h*w)
        scipy.misc.imsave('./test%s/%s_truth.png'%(MODEL_INDEX,filename),truth[0,:,:] /21 )
        scipy.misc.imsave('./test%s/%s_pred.png'%(MODEL_INDEX,filename), pred[0,:,:] / 21 )
        loss.append(loss_val)
        print('Iteration: %s'%str(i) + ' | Filename: %s'%filename + ' | Error rate: %s'%str(loss_val))
    np.save('./models/tstAccuracy%s'%MODEL_INDEX, np.array(loss))
    print('='*40)


if __name__=='__main__':
    if not os.path.exists('./test%s'%MODEL_INDEX):
        os.makedirs('./test%s'%MODEL_INDEX)
    tf.app.run(main=main, argv=[])
