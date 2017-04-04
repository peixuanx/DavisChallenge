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
from PIL import Image

from config import *

def onlineTraining():
    
    # read current file index
    file_index = 0
    if os.path.exists('./file_index'):
        f = open('./file_index')
        for line in f:
            file_index = int(line)
        f.close()
    f = open('./file_index', 'a+')
    
    # import data
    davis_reader = read_davis.DavisReader()
    
    # Create the model
    fcn = Davis_FCN.FCN() 
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y_ = tf.placeholder(tf.float32)
    y = fcn.build(x, train=True, num_classes=NUM_CLASSES, debug=True)

	# Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    train_step = train_step.minimize(cross_entropy)

    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    saver = tf.train.Saver()
    if file_index != 0:
        saver.restore(sess, "./models/model%s"%MODEL_INDEX)
        print("Model restored ...")
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Training
    print('='*40)
    print('Online Training ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys = davis_reader.next_batch()
        _, loss_val = sess.run([train_step,cross_entropy], 
                                feed_dict={x: batch_xs, y_: batch_ys})
        save_path = saver.save(sess, "./models/model%s"%MODEL_INDEX)      
        loss.append(loss_val)
        np.save('./models/trCrossEntropyLoss%s'%MODEL_INDEX, np.array(loss))
        f.write(str(file_index+i+1)+'\n')
        log = 'Iteration: %s'%str(i) + \
                ' | Model saved in file: %s'%save_path + ' | Cross entropy loss: %s'%str(loss_val)
        print(log)
    f.close()
    #np.save('./models/trCrossEntropyLoss%s'%MODEL_INDEX, np.array(loss))


def main(argv):
	
    # import data
    davis_reader = read_davis.DavisReader()

    # Create the model
    fcn = Davis_FCN.FCN()
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y_ = tf.placeholder(tf.float32) 
    y = fcn.build(x, num_classes=NUM_CLASSES, debug=False)

    # Define prediction
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_, 3))
    
    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    #saver = tf.train.Saver()
    saver = tf.train.import_meta_graph("./models/model%s.meta"%MODEL_INDEX)    
    saver.restore(sess, "./models/model%s"%MODEL_INDEX)
    print("Model restored ...")
    
    init = tf.global_variables_initializer()
    sess.run(init)

    # Testing
    print('='*40)
    print('Tresting ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys, filename = read_davis.next_test()
        err, truth, pred = sess.run([correct_prediction, tf.argmax(y_,3), tf.argmax(y,3)], 
                            feed_dict={x: batch_xs, y_: batch_ys})
        h,w = np.shape(err[0])
        loss_val = len(np.where(err[0]==False)[0])/(h*w)

        scipy.misc.imsave('./test%s/%s_truth.png'%(MODEL_INDEX,filename),
                            truth[0,:,:])
        scipy.misc.imsave('./test%s/%s_pred.png'%(MODEL_INDEX,filename), 
                            pred[0,:,:])
        loss.append(loss_val)
        print('Iteration: %s'%str(i) + ' | Filename: %s'%filename + ' | Error rate: %s'%str(loss_val))
    np.save('./models/tstAccuracy%s'%MODEL_INDEX, np.array(loss))
    print('='*40)


if __name__=='__main__':
    if not os.path.exists('./test%s'%MODEL_INDEX):
        os.makedirs('./test%s'%MODEL_INDEX)
    tf.app.run(main=main, argv=[])
