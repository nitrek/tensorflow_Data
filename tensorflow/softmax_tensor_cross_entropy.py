from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    print("loading training data")
    trainX = csv_to_numpy_array("trainX1.csv", delimiter="\t")
    trainY = csv_to_numpy_array("trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("testX1.csv", delimiter="\t")
    testY = csv_to_numpy_array("testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()


numFeatures = trainX.shape[1]

numLabels = trainY.shape[1]

numEpochs = 250

# learningRate = tf.train.exponential_decay(learning_rate=0.0008,
#                                           global_step= 1,
#                                           decay_steps=trainX.shape[0],
#                                           decay_rate= 0.95,
#                                           staircase=True)



X = tf.placeholder(tf.float32, [None, numFeatures])

Y = tf.placeholder(tf.float32, [None, numLabels])

tf.set_random_seed(100)

weights = tf.Variable(tf.random_normal([numFeatures,numLabels]))

bias = tf.Variable(tf.random_normal([numLabels]))



prediction = tf.matmul(X, weights) + bias

activation_OP = tf.nn.softmax_cross_entropy_with_logits(prediction , Y)

cost_OP = tf.reduce_mean(activation_OP)

training_OP = tf.train.AdamOptimizer().minimize(cost_OP)

init_OP = tf.initialize_all_variables()

epoch_values=[]
accuracy_values=[]
cost_values=[]

sess = tf.Session()
sess.run(init_OP)

# Training epochs
for i in range(numEpochs):
    step, c = sess.run([training_OP,cost_OP], feed_dict={X: trainX, Y: trainY})
    print   '.',

correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

print 'Accuracy:',accuracy.eval({X:testX, Y:testY}) 

saver = tf.train.Saver()
saver.save(sess, "traineSd_variables_cross_entropy.ckpt")
sess.close()
