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
    davis_reader = read_davis.DavisReader(currentTrainImageId=file_index, mode='video')

    # Create the model
    fcn = Davis_FCN.FCN() 
    x = tf.placeholder(tf.float32) #shape=[batch size, dimemsionality] 
    y_ = tf.placeholder(tf.float32)
    
    # imbalanced weighted
    pixels = tf.reduce_sum(y_, [1,2])
    pixels_total = tf.expand_dims(tf.reduce_sum(pixels,1),1)
    ratio = tf.divide(pixels, pixels_total)
    ratio = tf.expand_dims(ratio, 1)
    ratio = tf.expand_dims(ratio, 1)
    y_weighted = tf.multiply(y_, ratio)


    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))

    y = fcn.build(x, train=True, num_classes=NUM_CLASSES,
                    random_init_fc8=True, debug=True)

    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_weighted, logits=y))
    #train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    train_step = train_step.minimize(cross_entropy)

    # Define prediction
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_, 3))
    
    # Session Define
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    if not os.path.exists('models_online'):
        os.makedirs('models_online')
        saver.restore(sess, "./models/model11")
        print("Offline Model restored ...")
    elif file_index ==0:
        saver.restore(sess, "./models_online/model%s"%str(MODEL_INDEX))
        print("Online Model%s restored ..."%str(MODEL_INDEX)) 
    else:
        saver.restore(sess, "./models_online/model%s"%str(MODEL_INDEX-1))
        print("Online Model%s restored ..."%str(MODEL_INDEX-1)) 
    
    # Training
    print('='*40)
    print('Online Training ...')
    loss = []
    for i in range(MAX_ITER):
        batch_xs, batch_ys = davis_reader.next_batch()
        xs = np.zeros((1,)+batch_xs[0].shape)
        ys = np.zeros((1,)+batch_ys[0].shape)
        for n in range(davis_reader.videoSize):
            xs[0]=batch_xs[n]
            ys[0]=batch_ys[n]
            _, loss_val = sess.run([train_step,cross_entropy], 
                            feed_dict={x: xs, y_: ys})
            loss.append(loss_val)
        save_path = saver.save(sess, "./models_online/model%s"%MODEL_INDEX)      
        np.save('./models_online/trCrossEntropyLoss%s'%MODEL_INDEX, np.array(loss))
        f.write(str(file_index+i+1)+'\n')
        log = 'Iteration: %s'%str(i) + \
            ' | Model saved in file: %s'%save_path + ' | Cross entropy loss: %s'%str(loss_val)
        print(log)

        for n in range(davis_reader.videoSize):
            xs[0]=batch_xs[n]
            ys[0]=batch_ys[n]
            err, truth, pred = sess.run([correct_prediction, tf.argmax(y_,3), tf.argmax(y,3)],
                                feed_dict={x: xs, y_: ys})
            h,w = np.shape(err[0])
            loss_val_test = len(np.where(err[0]==False)[0])/(h*w)
            print('Video: %s'%file_index + ' | Error rate: %s'%str(loss_val))
            scipy.misc.imsave('./online%s/%s_truth_%s.png'%(MODEL_INDEX,str(i),str(n)),
                            truth[0,:,:]*255)
            scipy.misc.imsave('./online%s/%s_pred_%s.png'%(MODEL_INDEX,str(i),str(n)),
                            pred[0,:,:]*255)
            loss_test.append(loss_val_test)
    f.close()

if __name__=='__main__':
    if not os.path.exists('./online%s'%MODEL_INDEX):
        os.makedirs('./online%s'%MODEL_INDEX)
    tf.app.run(main=main, argv=[])
